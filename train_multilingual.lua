
require '.'
require 'shortcut'
require 'SelectNetRich'
require 'DepRichDataIter'

local MultiTrainer = torch.class('MultilingualTrainer')

function MultiTrainer:showOpts()
  local tmp_vocab = self.opts.vocab
  self.opts.vocab = nil
  print(self.opts)
  self.opts.vocab = tmp_vocab
end

function MultiTrainer:train()
  local dataIter = DepRichDataIter.createBatchMultiShuffleSort(self.train_multi_sents, self.opts.vocab, self.opts.batchSize, self.opts.feats, true)
  local dataSize = self.trainSize
  local curDataSize = 0
  local percent, inc = 0.001, 0.001
  local timer = torch.Timer()
  local sgdParam = self.opts.sgdParam
  local cnt = 0
  local totalLoss = 0
  local totalCnt = 0
  
  for x, x_mask, x_feats, y in dataIter do
    local loss = self.net:trainBatch(x, x_mask, x_feats, y, sgdParam)
    local y_mask = x_mask[{ {2, -1}, {} }]
    
    totalLoss = totalLoss + loss * x:size(2)
    totalCnt = totalCnt + y_mask:sum()
    
    curDataSize = curDataSize + x:size(2)
    local ratio = curDataSize / dataSize
    if ratio >= percent then
      local wps = totalCnt / timer:time().real
      xprint( '\repoch %d %.3f %.4f (%s) / %.2f wps ... ', self.iepoch, ratio, totalLoss/totalCnt, readableTime(timer:time().real), wps )
      percent = math.floor(ratio / inc) * inc
      percent = percent + inc
    end
    
    cnt = cnt + 1
    if cnt % 5 == 0 then
      collectgarbage()
    end
  end
  
  local nll = totalLoss / totalCnt
  return nll, math.exp(nll)
end

function MultiTrainer:valid(validFile)
  local dataIter = DepRichDataIter.createBatchSort(self.opts.vocab, validFile, self.opts.batchSize, self.opts.maxTrainLen)
  local totalCnt = 0
  local totalLoss = 0
  local cnt = 0
  local UAS_c, UAS_t = 0, 0
  for x, x_mask, x_pos, y in dataIter do
    local loss, y_preds = self.net:validBatch(x, x_mask, x_pos, y)
    
    totalLoss = totalLoss + loss * x:size(2)
    local y_mask = x_mask[{ {2, -1}, {} }]
    
    local y_p = torch.LongTensor(y:size(1), y:size(2))
    for t = 2, x:size(1) do
      local _, mi = y_preds[t]:max(2)
      if self.opts.useGPU then mi = mi:double() end
      y_p[{ t-1, {} }] = mi
    end
    UAS_c = UAS_c + y:eq(y_p):double():cmul(y_mask):sum()
    UAS_t = UAS_t + y_mask:sum()
    
    totalCnt = totalCnt + y_mask:sum()
    cnt = cnt + 1
    if cnt % 5 == 0 then
      collectgarbage()
    end
  end
  
  local entropy = totalLoss / totalCnt
  local ppl =  torch.exp(entropy)
  
  return {entropy = entropy, ppl = ppl, UAS = (UAS_c / UAS_t)}
end

function MultiTrainer:validConllx(validFile)
  local dataIter = DepRichDataIter.createBatch(self.opts.vocab, validFile, self.opts.batchSize, self.opts.feats, self.opts.maxTrainLen)
  local totalCnt = 0
  local totalLoss = 0
  local cnt = 0
  
  local sents_dep = {}
  local y_tmp = torch.LongTensor(self.opts.maxTrainLen, self.opts.batchSize)
  for x, x_mask, x_feats, y in dataIter do
    local loss, y_preds = self.net:validBatch(x, x_mask, x_feats, y)
    totalLoss = totalLoss + loss * x:size(2)
    local y_mask = x_mask[{ {2, -1}, {} }]
    
    local y_p = y_tmp:resize(y:size(1), y:size(2))
    for t = 2, x:size(1) do
      local _, mi = y_preds[t]:max(2)
      if self.opts.useGPU then mi = mi:double() end
      y_p[{ t-1, {} }] = mi
    end
    
    for i = 1, y_mask:size(2) do
      local slen = y_mask[{ {}, i }]:sum()
      local sent_dep = {}
      for j = 1, slen do
        sent_dep[#sent_dep + 1] = y_p[{ j, i }] - 1
      end
      sents_dep[#sents_dep + 1] = sent_dep
    end
    
    totalCnt = totalCnt + y_mask:sum()
    cnt = cnt + 1
    if cnt % 5 == 0 then
      collectgarbage()
    end
  end
  
  local dep_iter = DepRichDataIter.conllx_iter(validFile)
  local sent_idx = 0
  local tokens = validFile:splitc('/')
  local filename = tokens[#tokens]
  local sys_out = self.opts.modelDir .. '/tmp/' .. filename .. '__tmp__.dep'
  local fout = io.open(sys_out, 'w')
  for dsent in dep_iter do
    sent_idx = sent_idx + 1
    local sent_dep = sents_dep[sent_idx]
    assert(#sent_dep == #dsent)
    for i, ditem in ipairs(dsent) do
      -- 1  Influential _ JJ  JJ  _ 2 amod  _ _
      fout:write(string.format('%d\t%s\t_\t%s\t_\t%s\t%d\tN_A\t_\t_\n', ditem.p1, ditem.wd, ditem.pos, ditem.xfeats, sent_dep[i]))
    end
    fout:write('\n')
  end
  fout:close()
  
  local conllx_eval = self.opts.evalType == 'stanford' and require 'conllx_eval' or require 'conllx2006_eval'
  local LAS, UAS, noPunctLAS, noPunctUAS = conllx_eval.eval(sys_out, validFile)
  
  local entropy = totalLoss / totalCnt
  local ppl =  torch.exp(entropy)
  
  return {entropy = entropy, ppl = ppl, UAS = noPunctUAS}
end

function MultiTrainer:main()
  local model_opts = require 'multi_model_opts'
  local opts = model_opts.getOpts()
  self.opts = opts
  
  local train_dir = opts.train
  local valid_dir = opts.valid
  local train_files = listDir(train_dir)
  local valid_files = listDir(valid_dir)
  if opts.test and opts.test ~= '' then
    local test_dir = opts.test
    local test_files = listDir(test_dir)
  end

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
  print('Num lang: ' .. num_lang)
  if opts.batchSize ~= num_lang and num_lang > 1 then
    opts.batchSize = num_lang * opts.batchSize
  end
  print('Batch size: ' .. opts.batchSize)
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
    opts.vocab = DepRichDataIter.createMultiVocab(opts.train, opts.ignoreCase, opts.freqCut, opts.maxNVocab, opts.feats, uDpos, opts.modelType)
    torch.save(vocabPath, opts.vocab)
    xprintln('****create vocab from scratch****\n\n')
  end
  
  self.net = SelectNetRich(opts)
  self:showOpts()
  
  self.train_multi_sents = DepRichDataIter.loadMultiSents(opts.vocab, opts.train, opts.maxTrainLen, opts.feats, false)

  local bestUAS = 0
  local bestModel = torch.FloatTensor(self.net.params:size())
  local timer = torch.Timer()
  
  for epoch = 1, self.opts.maxEpoch do
    self.iepoch = epoch
    local startTime = timer:time().real
    local train_nll, train_perp = self:train()
    xprintln('\nepoch %d TRAIN %f (%f) ', epoch, train_nll, train_perp)
    
    print 'Valid Performance'
    local avgUAS = 0
    local counter = 0
    for _, f in ipairs(valid_files) do
      counter = counter + 1
      local name = f:match( "([^/]+)$" )
      local tokens = name:splitc('-')
      local langId = tokens[2]
      print('Language:', langId)
      print('File = ', f)
      local vret = self:validConllx(f)
      print(vret)
      avgUAS = avgUAS + vret.UAS
      print('Total UAS:', avgUAS)
    end
    avgUAS = avgUAS / counter
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
          print('File = ', f)
          local vret = self:validConllx(f)
          print(vret)
        end
      end
    else
      if not opts.disableEearlyStopping then
        xprintln('UAS on valid not increase! early stopping!')
        break
      end
    end
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
    print('File = ', f)
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
      print('File = ', f)
      local vret = self:validConllx(f)
      print(vret)
    end
  end
  
end

local function main()
  local trainer = MultilingualTrainer()
  trainer:main()
end

if not package.loaded['train_multilingual'] then
  main()
else
  print '[train_multilingual] loaded as package!'
end


