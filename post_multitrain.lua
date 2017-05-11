
require '.'
require 'shortcut'
require 'SelectNetRich'
require 'train_multilingual'

local Trainer = torch.class('MultiPostTrainer', 'MultilingualTrainer')

local function getOpts()
  local cmd = torch.CmdLine()
  cmd:option('--load', '', 'model path')
  cmd:option('--save', 'model.t7', 'save model path')
  cmd:option('--lr', 0.001, 'learning rate')
  cmd:option('--maxEpoch', 30, 'maximum number of epochs')
  cmd:option('--optimMethod', 'SGD', 'optimization algorithm')
  cmd:option('--decay', 1, 'decay learning rate')
  
  local opts = cmd:parse(arg)
  
  return opts
end

function Trainer:main()
  local opts_ = getOpts()
  self.opts = torch.load( opts_.load:sub(1, -3) .. 'state.t7' )
  assert(self.opts.save ~= opts_.save)
  self.opts.load = opts_.load
  self.opts.save = opts_.save
  self.opts.lr = opts_.lr 
  self.opts.maxEpoch = opts_.maxEpoch
  self.opts.optimMethod = opts_.optimMethod
  local opts = self.opts
  
  torch.manualSeed(opts.seed + 1)
  if opts.useGPU then
    require 'cutorch'
    require 'cunn'
    cutorch.manualSeed(opts.seed + 1)
  end

  local train_dir = opts.train
  local valid_dir = opts.valid
  local test_dir = opts.test
  local train_files = listDir(train_dir)
  local valid_files = listDir(valid_dir)
  local test_files = listDir(test_dir)

  -- collect statistics from data
  xprintln('\n********Training stats********')
  
  local totalTrainSize = 0
  local minTrainSize = 1000000
  local smallest_train_set = ''
  local num_lang = 0
  for _, f in ipairs(train_files) do
    local size = unpack(DepRichDataIter.getDataSize({f}))
    local name = f:match( "([^/]+)$" )
    local tokens = name:splitc('-')
    local langId = tokens[2]
    totalTrainSize = totalTrainSize + size
    num_lang = num_lang + 1
    if size < minTrainSize then
      minTrainSize = size
      smallest_train_set = langId
    end
  end
  
  self.trainSize = num_lang * minTrainSize
  self.num_lang = num_lang
  self.opts = opts

  xprintln('Total training sentences\t: %d sents', totalTrainSize)
  xprintln('Smallest treebank\t\t: %s, %d sents', smallest_train_set, minTrainSize)
  xprintln('\n********Valid stats********')
  
  local minValSize = 1000000
  local smallest_valid_set = ''
  for _, f in ipairs(valid_files) do
    local size = unpack(DepRichDataIter.getDataSize({f}))
    local name = f:match( "([^/]+)$" )
    local tokens = name:splitc('-')
    local langId = tokens[2]
    if size < minValSize then
      minValSize = size
      smallest_valid_set = langId
    end
  end
  
  xprintln('Smallest treebank\t\t: %s, %d sents\n', smallest_valid_set, minValSize)
  
  -- local vocabPath = opts.train .. '.tmp.vocab.t7'
  local vocabPath = opts.vocabFile
  local uDpos = opts.uDVocab .. '/ud_pos.vocab'
  
  local recreateVocab = true
  if paths.filep(vocabPath) then
    opts.vocab = torch.load(vocabPath)
    if opts.vocab.ignoreCase == opts.ignoreCase and opts.vocab.freqCut == opts.freqCut and opts.vocab.maxNVocab == opts.maxNVocab then
      recreateVocab = false
      DepRichDataIter.showVocab(opts.vocab)
      print '****load from existing vocab!!!****\n\n'
    end
  end
  if recreateVocab then
    opts.vocab = DepRichDataIter.createMultiVocab(opts.train, opts.ignoreCase, opts.freqCut, opts.maxNVocab, opts.feats, uDpos)
    torch.save(vocabPath, opts.vocab)
    xprintln('****create vocab from scratch****\n\n')
  end
  
  self.net = SelectNetRich(opts)
  self:showOpts()
  
  xprintln('load from %s ...', opts.load)
  self.net:load(opts.load)
  xprintln('load from %s done!', opts.load)
  
  self.train_multi_sents = DepRichDataIter.loadMultiSents(opts.vocab, opts.train, opts.maxTrainLen, opts.feats, false)
  local bestUAS = 0
  local bestModel = torch.FloatTensor(self.net.params:size())
  local timer = torch.Timer()
  
  self.opts.sgdParam = {learningRate = opts.lr}

  local avgUAS = 0
  for _, f in ipairs(valid_files) do
    local name = f:match( "([^/]+)$" )
    local tokens = name:splitc('-')
    local langId = tokens[2]
    print('Language:', langId)
    local vret = self:validConllx(f)
    print(vret)
    avgUAS = avgUAS + vret.UAS
    print('Total UAS:', avgUAS)
  end

  bestUAS = avgUAS / num_lang
  print('Avg UAS:', bestUAS)
  self.net:getModel(bestModel)
  
  for epoch = 1, self.opts.maxEpoch do
    self.iepoch = epoch
    local startTime = timer:time().real
    
    local train_nll, train_perp = self:train()
    xprintln('\nepoch %d TRAIN %f (%f) ', epoch, train_nll, train_perp)

    print 'Valid Performance'
    local avgUAS = 0
    for _, f in ipairs(valid_files) do
      local name = f:match( "([^/]+)$" )
      local tokens = name:splitc('-')
      local langId = tokens[2]
      print('Language:', langId)
      local vret = self:validConllx(f)
      print(vret)
      avgUAS = avgUAS + vret.UAS
      print('Total UAS:', avgUAS)
    end
    avgUAS = avgUAS / num_lang
    xprintln('Average UAS: %f', avgUAS)

    local endTime = timer:time().real
    xprintln('time spend %s', readableTime(endTime - startTime))
    
    if bestUAS < avgUAS then
      bestUAS = avgUAS
      self.net:getModel(bestModel)
      if opts.test and opts.test ~= '' then
        print 'Test Performance'
        for _, f in ipairs(test_files) do
          local name = f:match( "([^/]+)$" )
          local tokens = name:splitc('-')
          local langId = tokens[2]
          print('Language:', langId)
          local vret = self:validConllx(f)
          print(vret)
        end
      end
    else
      xprintln('UAS on valid not increase! early stopping!')
      break
    end
    
    self.opts.sgdParam.learningRate = self.opts.sgdParam.learningRate * opts_.decay
  end
  
  -- save final model
  self.net:setModel(bestModel)
  opts.sgdParam = nil
  self.net:save(opts.save, true)
  xprintln('model saved at %s', opts.save)
  
  -- show final perform
  print 'Final Valid Performance'
  for _, f in ipairs(valid_files) do
    local name = f:match( "([^/]+)$" )
    local tokens = name:splitc('-')
    local langId = tokens[2]
    print('Language:', langId)
    local vret = self:validConllx(f)
    print(vret)
  end
  
  if opts.test and opts.test ~= '' then
    print 'Final Test Performance'
    for _, f in ipairs(test_files) do
      local name = f:match( "([^/]+)$" )
      local tokens = name:splitc('-')
      local langId = tokens[2]
      print('Language:', langId)
      local vret = self:validConllx(f)
      print(vret)
    end
  end
  
end

local function main()
  local trainer = MultiPostTrainer()
  trainer:main()
end

if not package.loaded['post_multitrain'] then
  main()
else
  print '[post_multitrain] loaded as package!'
end

