
require '.'
require 'shortcut'
require 'SelectNetRich'
require 'DepRichDataIter'
require 'hdf5'
require 'MLP'

local LabeledMultiTrainer = torch.class('LabeledMultiModelTrainer')

local function getOpts()
  local cmd = torch.CmdLine()
  cmd:option('--mode', 'train', 'two modes: [generate] generate training data; [train] train labeled model')
  cmd:option('--modelPath', '', 'model path')
  cmd:option('--outTrainDataPath', '', 'where will you save the training data')
  cmd:option('--inTrain', '', 'input training data path')
  cmd:option('--inValid', '', 'input validation data path')
  cmd:option('--inTest', '', 'input test data path')
  cmd:option('--outValid', '', 'valid conllx file from last step')
  cmd:option('--outTest', '', 'test conllx file from last step')
  cmd:option('--uDVocab', '', 'UD vocab path')
  cmd:option('--language', 'Other', 'English or Chinese or Other')
  
  cmd:text('')
  cmd:text('==Options for MLP==')
  cmd:option('--seed', 123, 'random seed')
  cmd:option('--useGPU', false, 'use gpu')
  cmd:option('--snhids', '1460,400,400,45', 'string hidden sizes for each layer')
  cmd:option('--ftype', '|x|', 'type: x, xe, xpe. For example: |x|xe|xpe|')
  cmd:option('--activ', 'relu', 'options: tanh, relu')
  cmd:option('--dropout', 0, 'dropout rate (dropping)')
  cmd:option('--inDropout', 0, 'dropout rate (dropping)')
  cmd:option('--batchNorm', false, 'add batch normalization')
  cmd:option('--maxEpoch', 10, 'max number of epochs')
  cmd:option('--dataset', '', 'dataset')
  cmd:option('--batchSize', 256, '')
  cmd:option('--lr', 0.01, '')
  cmd:option('--optimMethod', 'AdaGrad', 'options: SGD, AdaGrad, Adam')
  cmd:option('--save', '', 'save path')
  
  local opts = cmd:parse(arg)
  
  return opts
end

function LabeledMultiTrainer:showOpts()
  local tmp_vocab = self.opts.vocab
  self.opts.vocab = nil
  print(self.opts)
  self.opts.vocab = tmp_vocab
end

function LabeledMultiTrainer:validConllx(validFile, outputConllFile, defaultLabel)
  xprintln('default label is %s', defaultLabel)
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
    if cnt % 100 == 0 then
      collectgarbage()
    end
  end
  
  local dep_iter = DepRichDataIter.conllx_iter(validFile)
  local sent_idx = 0
  local sys_out = outputConllFile
  local fout = io.open(sys_out, 'w')
  for dsent in dep_iter do
    sent_idx = sent_idx + 1
    local sent_dep = sents_dep[sent_idx]
    assert(#sent_dep == #dsent)
    for i, ditem in ipairs(dsent) do
      -- 1  Influential _ JJ  JJ  _ 2 amod  _ _
      fout:write(string.format('%d\t%s\t_\t%s\t_\t%s\t%d\t%s\t_\t_\n', ditem.p1, ditem.wd, ditem.pos, ditem.xfeats, sent_dep[i], defaultLabel))
    end
    fout:write('\n')
  end
  fout:close()
  
  -- local conllx_eval = require 'conllx_eval'
  if  self.opts.evalType == nil then
    self.opts.evalType = 'stanford'
  end
  xprintln('eval type = %s', self.opts.evalType)
  local conllx_eval = self.opts.evalType == 'stanford' and require 'conllx_eval' or require 'conllx2006_eval'
  local LAS, UAS, noPunctLAS, noPunctUAS = conllx_eval.eval(sys_out, validFile)
  
  local entropy = totalLoss / totalCnt
  local ppl =  torch.exp(entropy)
  
  return {entropy = entropy, ppl = ppl, UAS = noPunctUAS}
end

function LabeledMultiTrainer:load(model_path)
  local opts = torch.load( model_path:sub(1, -3) .. 'state.t7' )
  self.opts = opts
  torch.manualSeed(opts.seed)
  if opts.useGPU then
    require 'cutorch'
    require 'cunn'
    cutorch.manualSeed(opts.seed)
  end
  
  assert(opts.vocab ~= nil, 'We must have an existing vocabulary!')
  self.net = SelectNetRich(opts)
  self:showOpts()
  print(self.opts.batchSize)
  xprintln('load from %s ...', model_path)
  self.net:load(model_path)
  xprintln('load from %s done!', model_path)
end

function LabeledMultiTrainer:createTrainData(data_paths, outDataPath, language, uDVocab)
  
  local uDrel = uDVocab .. '/ud_rel.vocab'
  local h5out = hdf5.open(outDataPath, 'w')

  self.rel_vocab = DepRichDataIter.createDepRelMultiVocab(data_paths.train, uDrel)
  print(self.rel_vocab)

  local function generateSplit(slabel, infile, batchSize, maxlen, feats, feat_dims)
    local gxdata = string.format('/%s/x', slabel)
    local gydata = string.format('/%s/y', slabel)
    local gxedata = string.format('/%s/xe', slabel)
    
    local gxfeatdata = {}
    local xfeatOpt = {}
    local we_dim = 0
    for i = 1, #feats do
      if feats[i] ~= 'we' then
        gxfeatdata[feats[i]] = string.format('/%s/x' .. feats[i], slabel)
        xfeatOpt[feats[i]] = hdf5.DataSetOptions()
        xfeatOpt[feats[i]]:setChunked(1024*10, feat_dims[i] * 2)
        xfeatOpt[feats[i]]:setDeflate()
      else
        we_dim = feat_dims[i]
      end
    end
    
    local xOpt = hdf5.DataSetOptions()
    xOpt:setChunked(1024*10, self.opts.nhid * 4)
    xOpt:setDeflate()
    
    local xeOpt = hdf5.DataSetOptions()
    xeOpt:setChunked(1024*10, we_dim * 2)
    xeOpt:setDeflate()
    
    local yOpt = hdf5.DataSetOptions()
    yOpt:setChunked(1024*10)
    yOpt:setDeflate()
    
    local isFirst = true
    local cnt = 0
    local diter = {}
    if slabel == 'train' then
      print('Generating training data..')
      diter = DepRichDataIter.createBatchLabelMulti(self.opts.vocab, self.rel_vocab, infile, batchSize, maxlen, self.opts.feats)
    else
      print('Generating valid/test data..')
      print('Batch size: ' .. batchSize)
      diter = DepRichDataIter.createBatchLabel(self.opts.vocab, self.rel_vocab, infile, batchSize, maxlen, self.opts.feats)
    end

    for x, x_mask, x_feats, y, sent_rels, sent_ori_rels in diter do
      self.net:validBatch(x, x_mask, x_feats, y)
      local dsize = x_mask:sum() - x_mask:size(2)

      assert(dsize == y:ne(0):sum(), 'size should be the same')
      local x_input = torch.zeros(dsize, self.opts.nhid * 4):float()
      local y_output = torch.zeros(dsize):int()
      local x_input_emb = torch.zeros(dsize, we_dim * 2):float()
      local x_input_feat_emb = {}
      for i = 1, #feats do
        if feats[i] ~= 'we' then
          x_input_feat_emb[feats[i]] = torch.zeros(dsize, feat_dims[i] * 2):float()
        end
      end
      
      local x_emb = self.net.mod_map.forward_lookup:forward(x)
      local x_feat_emb = {}
      for i, feat_vec in ipairs(x_feats) do
        -- feats contains we, so i-1
        local fwd_lookup = 'forward_' .. feats[i+1] .. '_lookup'
        x_feat_emb[feats[i+1]] = self.net.mod_map[fwd_lookup]:forward(feat_vec)
      end
      
      local example_cnt = 0
      for i, sent_rel in ipairs(sent_rels) do
        assert(x_mask[{ {}, i }]:sum() == #sent_rel + 1, 'MUST be the same length')
        for j, rel_id in ipairs(sent_rel) do
          local cur_id = j + 1
          local parent_id = y[{ j, i }]
          local cur_a = self.net.all_fwd_bak_hs[{ i, cur_id, {} }]
          local parent_a = self.net.all_fwd_bak_hs[{ i, parent_id, {} }]
          example_cnt = example_cnt + 1
          x_input[{ example_cnt, {1, 2 * self.opts.nhid} }] = cur_a:float()
          x_input[{ example_cnt, {2 * self.opts.nhid + 1, 4 * self.opts.nhid} }] = parent_a:float()
          y_output[{ example_cnt }] = rel_id
          
          local cur_emb = x_emb[{ cur_id, i, {} }]
          local parent_emb = x_emb[{ parent_id, i, {} }]
          x_input_emb[{ example_cnt, {1, we_dim} }] = cur_emb:float()
          x_input_emb[{ example_cnt, {we_dim + 1, 2 * we_dim} }] = parent_emb:float()

          local cur_feat_emb = {}
          local parent_feat_emb = {}
          for i = 1, #feats do
            if feats[i] ~= 'we' then
              cur_feat_emb[feats[i]] = x_feat_emb[feats[i]][{ cur_id, i, {} }]
              parent_feat_emb[feats[i]] = x_feat_emb[feats[i]][{ parent_id, i, {} }]
              x_input_feat_emb[feats[i]][{ example_cnt, {1, feat_dims[i]} }] = cur_feat_emb[feats[i]]:float()
              x_input_feat_emb[feats[i]][{ example_cnt, {feat_dims[i] + 1, 2 * feat_dims[i]} }] = parent_feat_emb[feats[i]]:float()
            end
          end
        end
      end
      
      if isFirst then
        h5out:write(gxdata, x_input, xOpt)
        h5out:write(gydata, y_output, yOpt)
        h5out:write(gxedata, x_input_emb, xeOpt)
        for i = 1, #feats do
          if feats[i] ~= 'we' then
            h5out:write(gxfeatdata[feats[i]], x_input_feat_emb[feats[i]], xfeatOpt[feats[i]])
          end
        end
        isFirst = false
      else
        h5out:append(gxdata, x_input, xOpt)
        h5out:append(gydata, y_output, yOpt)
        h5out:append(gxedata, x_input_emb, xeOpt)
        for i = 1, #feats do
          if feats[i] ~= 'we' then
            h5out:append(gxfeatdata[feats[i]], x_input_feat_emb[feats[i]], xfeatOpt[feats[i]])
          end
        end
      end
      
      cnt = cnt + 1
      if cnt % 100 == 0 then
        collectgarbage()
      end
      
      if cnt % 10 == 0 then
        xprint('cnt = %d\n', cnt)
      end
    end
    
    print( 'totally ' .. cnt )
  end

  local feats = {}
  local feat_dims = {}
  for _, token in ipairs(self.opts.feats:splitc(',')) do
    feats[#feats + 1] = token 
    if token == 'we' then
      we_idx = #feats
    end
  end
  for _, token in ipairs(self.opts.feat_dims:splitc(',')) do
    feat_dims[#feat_dims + 1] = token
    if we_idx == #feat_dims then
      we_dim = token
    end
  end
  assert(#feats == #feat_dims, 'Number of features and dims should be the same')
  
  local dlabel = self.rel_vocab.idx2rel[1]
  xprintln('the default dependency label is %s\n', dlabel)

  local maxTrainLen = 150
  if self.opts.maxTrainLen ~= nil then
    maxTrainLen = self.opts.maxTrainLen
    print('maxTrainLen = ', maxTrainLen)
  end

  local replaceField = require 'replace_conllx_field'
  local predictValidFiles = {}
  local predictTestFiles = {}

  local lang = {}
  local train_files = listDir(data_paths.train)
  local valid_files = listDir(data_paths.valid)
  for _, f in ipairs(valid_files) do
    local name = f:match( "([^/]+)$" )
    local tokens = name:splitc('-')
    local lang_id = tokens[2]
    lang[#lang + 1] = lang_id
    outfile = outDataPath .. '.' .. lang_id
    predictValidFiles[lang_id] = outfile .. '.valid.conllx'
    outvalid = data_paths.outvalid .. '-' .. lang_id ..'.dep'
    replaceField.replace(outvalid, predictValidFiles[lang_id], 8, dlabel)
    xprintln('change field 8 to %s', dlabel)
  end

  if self.opts.test and self.opts.test ~= '' then
    for _, f in ipairs(valid_files) do
      local name = f:match( "([^/]+)$" )
      local tokens = name:splitc('-')
      local lang_id = tokens[2]
      lang[#lang + 1] = lang_id
      outfile = outDataPath .. '.' .. lang_id
      predictTestFiles[lang_id] = outfile .. '.test.conllx'
      outtest = data_paths.outtest .. '-' .. lang_id ..'.dep'
      replaceField.replace(outtest, predictTestFiles[lang_id], 8, dlabel)
      xprintln('change field 8 to %s', dlabel)
    end
  end

  local num_lang = #train_files
  

  if  self.opts.evalType == nil then
      self.opts.evalType = 'stanford'
  end
  xprintln('eval type = %s', self.opts.evalType)
  local conllx_eval = self.opts.evalType == 'stanford' and require 'conllx_eval' or require 'conllx2006_eval'
  
  -- assert(language == 'Other')
  -- local valid_files = listDir(data_paths.valid)
  for _, f in ipairs(valid_files) do
    local name = f:match( "([^/]+)$" )
    local tokens = name:splitc('-')
    local lang_id = tokens[2]
    lang[#lang + 1] = lang_id
    print('Valid file:', lang_id)
    -- self:validConllx(f, predictValidFiles[lang_id], dlabel)
    print '===Valid==='
    conllx_eval.eval(predictValidFiles[lang_id], f)
    predictvalidsplit = 'predict_valid_' .. lang_id
    validsplit = 'valid_' .. lang_id
    generateSplit(predictvalidsplit, predictValidFiles[lang_id], self.opts.batchSize, 999999, feats, feat_dims)
    generateSplit(validsplit, f, self.opts.batchSize, 999999, feats, feat_dims)
  end
  if self.opts.test and self.opts.test ~= '' then
    local test_files = listDir(data_paths.test)
    for _, f in ipairs(test_files) do
      local name = f:match( "([^/]+)$" )
      local tokens = name:splitc('-')
      local lang_id = tokens[2]
      print('Test file:', lang_id)
      print '===Test==='
      conllx_eval.eval(predictTestFiles[lang_id], f)
      -- print(predictTestFiles[lang_id])
      -- self:validConllx(f, predictTestFiles[lang_id], dlabel)
      predicttestsplit = 'predict_test_' .. lang_id
      testsplit = 'test_' .. lang_id
      generateSplit(predicttestsplit, predictTestFiles[lang_id], self.opts.batchSize, 999999, feats, feat_dims)
      generateSplit(testsplit, f, self.opts.batchSize, 999999, feats, feat_dims)
    end
  end
  generateSplit('train', data_paths.train, self.opts.batchSize, maxTrainLen, feats, feat_dims)
  h5out:close()
end

local DataIter = {}
function DataIter.getNExamples(dataPath, label)
  local h5in = hdf5.open(dataPath, 'r')
  local x_data = h5in:read(string.format('/%s/x', label))
  local N = x_data:dataspaceSize()[1]
  return N
end

function DataIter.createBatch(dataPath, label, batchSize, ftype, feats)
  local h5in = hdf5.open(dataPath, 'r')
  
  local x_data = h5in:read(string.format('/%s/x', label))
  local xe_data = h5in:read(string.format('/%s/xe', label))
  
  local xfeat_data = {}
  for i = 1, #feats do
    if feats[i] ~= 'we' then
      xfeat_data[feats[i]] = h5in:read(string.format('/%s/x' .. feats[i], label))
    end
  end
  
  local y_data = h5in:read(string.format('/%s/y', label))
  local N = x_data:dataspaceSize()[1]
  local x_width = x_data:dataspaceSize()[2]
  local xe_width = xe_data:dataspaceSize()[2]
  local xfeat_width = {}
  for feat, xdata in pairs(xfeat_data) do
    xfeat_width[feat] = xdata:dataspaceSize()[2]
  end
  
  local istart = 1
  
  return function()
    if istart <= N then
      local iend = math.min(istart + batchSize - 1, N)
      local y = y_data:partial({istart, iend})
      
      local widths = {x_width}
      local xdatas = {x_data}
      if ftype:find('|xe|') then
        widths[#widths + 1] = xe_width
        xdatas[#xdatas + 1] = xe_data
      end
      if ftype:find('|xfeat|') then
        for i = 1, #feats do
          if feats[i] ~= 'we' then
            widths[#widths + 1] = xfeat_width[feats[i]]
            xdatas[#xdatas + 1] = xfeat_data[feats[i]]
          end
        end
      end
      
      local width = 0
      for _, w in ipairs(widths) do width = width + w end
      local x = torch.zeros(y:size(1), width):float()
      local s = 0
      for i, w in ipairs(widths) do
        x[{ {}, {s + 1, s + w} }] = xdatas[i]:partial({istart, iend}, {1, w})
        s = s + w
      end
      
      istart = iend + 1
      
      return x, y
    else
      h5in:close()
    end
  end
end


local RndBatcher = torch.class('RandomBatcher')
function RndBatcher:__init(h5in, x_data, xe_data, xfeat_data, y_data, bufSize, ftype)
  self.h5in = h5in
  self.x_data = x_data
  self.xe_data = xe_data
  self.xfeat_data = xfeat_data
  self.y_data = y_data
  self.bufSize = bufSize
  self.N = x_data:dataspaceSize()[1]
  self.x_width = x_data:dataspaceSize()[2]
  self.xe_width = xe_data:dataspaceSize()[2]
  self.xfeat_width = {}
  for feat, data in pairs(self.xfeat_data) do
    self.xfeat_width[feat] = xfeat_data[feat]:dataspaceSize()[2]
  end
  
  self.ftype = ftype
  
  self.istart = 1
  self.idx_chunk = 1
  self.chunk_size = 0
end

function RndBatcher:nextChunk()
  if self.istart <= self.N then
    local iend = math.min( self.istart + self.bufSize - 1, self.N )
    self.x_chunk = self.x_data:partial({self.istart, iend}, {1, self.x_width})
    self.xe_chunk = self.xe_data:partial({self.istart, iend}, {1, self.xe_width})
    self.xfeat_chunk = {}
    for feat, xdata in pairs(self.xfeat_data) do
      self.xfeat_chunk[feat] = xdata:partial({self.istart, iend}, {1, self.xfeat_width[feat]})
    end
    
    self.y_chunk = self.y_data:partial({self.istart, iend})
    
    self.chunk_size = iend - self.istart + 1
    
    self.istart = iend + 1
    
    return true
  else
    return false
  end
end

function RndBatcher:nextBatch(batchSize)
  if self.idx_chunk > self.chunk_size then
    if self:nextChunk() then
      self.idx_chunk = 1
      self.idxs_chunk = torch.randperm(self.chunk_size):long()
    else
      return nil
    end
  end
  
  local iend = math.min( self.idx_chunk + batchSize - 1, self.chunk_size )
  local idxs = self.idxs_chunk[{ {self.idx_chunk, iend} }]
  
  local y = self.y_chunk:index(1, idxs)
  
  local xs = {}
  local widths = {}
  local width = 0
  if self.ftype:find('|x|') then
    local x = self.x_chunk:index(1, idxs)
    width = width + self.x_width
    widths[#widths + 1] = self.x_width
    xs[#xs + 1] = x
  end
  
  if self.ftype:find('|xe|') then
    local xe = self.xe_chunk:index(1, idxs)
    width = width + self.xe_width
    widths[#widths + 1] = self.xe_width
    xs[#xs + 1] = xe
  end
  
  if self.ftype:find('|xfeat|') then
    for feat, _ in pairs(self.xfeat_data) do
      xfeat = self.xfeat_chunk[feat]:index(1, idxs)
      width = width + self.xfeat_width[feat]
      widths[#widths + 1] = self.xfeat_width[feat]
      xs[#xs + 1] = xfeat
    end
  end
  
  local x_ = torch.zeros(y:size(1), width):float()
  local s = 0
  for i, w in ipairs(widths) do
    x_[{ {}, {s+1, s+w} }] = xs[i]
    s = s + w
  end
  
  self.idx_chunk = iend + 1
  
  return x_, y
end

function DataIter.createBatchShuffle(dataPath, label, batchSize, ftype, feats)
  local h5in = hdf5.open(dataPath, 'r')
  
  local x_data = h5in:read(string.format('/%s/x', label))
  local xe_data = h5in:read(string.format('/%s/xe', label))
  local xfeat_data = {}
  for i = 1, #feats do
    if feats[i] ~= 'we' then
      xfeat_data[feats[i]] = h5in:read(string.format('/%s/x' .. feats[i], label))
    end
  end
  local y_data = h5in:read(string.format('/%s/y', label))
  
  local bufSize = 1000 * batchSize
  local rnd_batcher = RandomBatcher(h5in, x_data, xe_data, xfeat_data, y_data, bufSize, ftype)
  
  return function()
    return rnd_batcher:nextBatch(batchSize)
  end
end

function LabeledMultiTrainer:train_label()
  local dataIter = DataIter.createBatchShuffle(self.classifier_opts.dataset, 'train', 
    self.classifier_opts.batchSize, self.classifier_opts.ftype, self.feats)
  
  local dataSize = DataIter.getNExamples(self.classifier_opts.dataset, 'train')
  local percent, inc = 0.001, 0.001
  local timer = torch.Timer()
  
  local sgdParam = self.classifier_opts.sgdParam
  local cnt = 0
  local totalLoss = 0
  local totalCnt = 0
  for x, y in dataIter do
    local loss = self.mlp:trainBatch(x, y, sgdParam)
    totalLoss = totalLoss + loss * x:size(1)
    totalCnt = totalCnt + x:size(1)
    
    local ratio = totalCnt / dataSize
    if ratio >= percent then
      local wps = totalCnt / timer:time().real
      xprint( '\repoch %d %.3f %.4f (%s) / %.2f wps ... ', self.iepoch, ratio, totalLoss/totalCnt, readableTime(timer:time().real), wps )
      percent = math.floor(ratio / inc) * inc
      percent = percent + inc
    end
    
    cnt = cnt + 1
    if cnt % 100 == 0 then
      collectgarbage()
    end
  end
  
  return totalLoss / totalCnt
end

function LabeledMultiTrainer:valid_label(label)
  local dataIter = DataIter.createBatch(self.classifier_opts.dataset, label, 
    self.classifier_opts.batchSize, self.classifier_opts.ftype, self.feats)
  
  local cnt = 0
  local correct, total = 0, 0
  for x, y in dataIter do
    local correct_, total_ = self.mlp:validBatch(x, y)
    correct = correct + correct_
    total = total + total_
    cnt = cnt + 1
    if cnt % 100 == 0 then collectgarbage() end
  end
  
  return correct, total
end

function LabeledMultiTrainer:valid_label_conllx(label, conllx_file, gold_file)
  local dataIter = DataIter.createBatch(self.classifier_opts.dataset, label, 
    self.classifier_opts.batchSize, self.classifier_opts.ftype, self.feats)
  
  local cnt = 0
  local correct, total = 0, 0
  local lbl_idxs = {}
  for x, y in dataIter do
    local correct_, total_, y_pred = self.mlp:validBatch(x, y)
    correct = correct + correct_
    total = total + total_
    cnt = cnt + 1
    if cnt % 100 == 0 then collectgarbage() end
    
    local y_pred_ = y_pred:view(-1)
    for i = 1, y_pred_:size(1) do
      lbl_idxs[#lbl_idxs + 1] = y_pred_[i]
    end
  end
  
  local ilbl = 0
  local conllx_file_out = conllx_file .. '.out'
  
  -- begin
  local dep_iter = DepRichDataIter.conllx_iter(conllx_file)
  local sys_out = conllx_file_out
  local fout = io.open(sys_out, 'w')
  for dsent in dep_iter do
    for _, ditem in ipairs(dsent) do
      -- 1  Influential _ JJ  JJ  _ 2 amod  _ _
      ilbl = ilbl + 1
      local lbl = self.rel_vocab.idx2rel[ lbl_idxs[ilbl] ]
      fout:write( string.format('%d\t%s\t_\t%s\t_\t%s\t%d\t%s\t_\t_\n', ditem.p1, ditem.wd, ditem.pos, ditem.xfeats, ditem.p2, lbl) )
    end
    fout:write('\n')
  end
  fout:close()
  -- end
  
  local conllx_eval
  -- xprintln('Language = %s', self.classifier_opts.language)
  if self.classifier_opts.language == 'Other' then
    conllx_eval = require 'conllx2006_eval'
  else
    conllx_eval = require 'conllx_eval'
  end
  
  local LAS, UAS, noPunctLAS, noPunctUAS = conllx_eval.eval(sys_out, gold_file)
  return {LAS = noPunctLAS, UAS = noPunctUAS}
end

function LabeledMultiTrainer:trainLabeledClassifier(opts)
  torch.manualSeed(opts.seed)
  if opts.useGPU then
    require 'cutorch'
    require 'cunn'
    cutorch.manualSeed(opts.seed)
  end

  local predictValidFiles = {}
  local predictTestFiles = {}
  local lang = {}
  local train_files = listDir(opts.inTrain)
  local valid_files = listDir(opts.inValid)
  if self.opts.test and self.opts.test ~= '' then
    local test_files = listDir(opts.inTest)
  end
  for _, f in ipairs(valid_files) do
    local name = f:match( "([^/]+)$" )
    local tokens = name:splitc('-')
    local lang_id = tokens[2]
    lang[#lang + 1] = lang_id
    prefix = opts.dataset .. '.' .. lang_id
    predictValidFiles[lang_id] = prefix .. '.valid.conllx'
  end
  if self.opts.test and self.opts.test ~= '' then
    local name = f:match( "([^/]+)$" )
    local tokens = name:splitc('-')
    local lang_id = tokens[2]
    lang[#lang + 1] = lang_id
    prefix = opts.dataset .. '.' .. lang_id
    predictTestFiles[lang_id] = prefix .. '.test.conllx'
  end
  print('Batch size:' .. opts.batchSize)

  local mlp = MLP(opts)
  opts.sgdParam = {learningRate = opts.lr}
  opts.curLR = opts.lr
  print(opts)
  
  self.classifier_opts = opts
  self.classifier_opts.batchSize = self.opts.batchSize * self.classifier_opts.batchSize
  print('New batch size: ' .. self.classifier_opts.batchSize)
  self.mlp = mlp
  
  local timer = torch.Timer()
  local bestAcc = 0
  local bestModel = torch.FloatTensor(mlp.params:size())
  local bestLAS = 0
  
  local uDrel = opts.uDVocab .. '/ud_rel.vocab'
  self.rel_vocab = DepRichDataIter.createDepRelMultiVocab(opts.inTrain, uDrel)
  opts.rel_vocab = self.rel_vocab
  xprintln('load rel_vocab done!')

  local feats = {}
  local feat_dims = {}
  for _, token in ipairs(self.opts.feats:splitc(',')) do
    feats[#feats + 1] = token 
    if token == 'we' then
      we_idx = #feats
    end
  end
  for _, token in ipairs(self.opts.feat_dims:splitc(',')) do
    feat_dims[#feat_dims + 1] = token
    if we_idx == #feat_dims then
      we_dim = token
    end
  end

  assert(#feats == #feat_dims, 'Number of features and dims should be the same')
  self.feats = feats
  self.feat_dims = feat_dims
  
  for epoch = 1, opts.maxEpoch do
    self.iepoch = epoch
    local startTime = timer:time().real
    local trainCost = self:train_label()
    xprintln('\repoch %d TRAIN nll %f ', epoch, trainCost)
    
    for _, lang_id in pairs(lang) do
      local validCor, validTot = self:valid_label('valid_' .. lang_id)
      local validAcc = validCor / validTot
      xprintln('VALID %s %d/%d = %f ', lang_id, validCor, validTot, validAcc)
    end
    local endTime = timer:time().real
    xprintln('lr = %.4g (%s)', opts.curLR, readableTime(endTime - startTime))
    
    local avgLAS = 0
    local counter = 0
    for _, f in ipairs(valid_files) do
      counter = counter + 1
      local name = f:match( "([^/]+)$" )
      local tokens = name:splitc('-')
      local lang_id = tokens[2]
      slabel = 'predict_valid_' .. lang_id
      predictValidFile = predictValidFiles[lang_id]
      print('Language = ', lang_id)
      print('File = ', f)
      local vret = self:valid_label_conllx(slabel, predictValidFile, f)
      print('==Valid Perf==')
      print(vret)
      print('\n')
      avgLAS = avgLAS + vret.LAS
    end
    avgLAS = avgLAS / counter
    print('Average LAS:', avgLAS)
    
    if avgLAS > bestLAS then
      bestLAS = avgLAS
      mlp:getModel(bestModel)
      if self.opts.test and self.opts.test ~= '' then
        for _, f in ipairs(test_files) do
          local name = f:match( "([^/]+)$" )
          local tokens = name:splitc('-')
          local lang_id = tokens[2]
          slabel = 'predict_test_' .. lang_id
          print('Language = ', lang_id)
          print('File = ', f)
          predictTestFile = predictTestFiles[lang_id]
          print(slabel, predictTestFile)
          local tret = self:valid_label_conllx(slabel, predictTestFile, f)
          print('==Test Perf==')
          print(tret)
          print('\n')
        end
      end
    end
  end
  
  mlp:setModel(bestModel)
  opts.sgdParam = nil
  mlp:save(opts.save, true)
  xprintln('model saved at %s', opts.save)

  for _, f in ipairs(valid_files) do
    local name = f:match( "([^/]+)$" )
    local tokens = name:splitc('-')
    local lang_id = tokens[2]
    slabel = 'predict_valid_' .. lang_id
    print('Language = ', lang_id)
    print('File = ', f)
    predictValidFile = predictValidFiles[lang_id]
    local vret = self:valid_label_conllx(slabel, predictValidFile, f)
    print('==Valid Perf==')
    print(vret)
    print('\n')
  end
  
  if self.opts.test and self.opts.test ~= '' then
    for _, f in ipairs(test_files) do
      local name = f:match( "([^/]+)$" )
      local tokens = name:splitc('-')
      local lang_id = tokens[2]
      slabel = 'predict_test_' .. lang_id
      print('Language = ', lang_id)
      print('File = ', f)
      predictTestFile = predictTestFiles[lang_id]
      local tret = self:valid_label_conllx(slabel, predictTestFile, f)
      print('==Test Perf==')
      print(tret)
      print('\n')
    end
  end
end

local function main()
  local opts = getOpts()
  local trainer = LabeledMultiModelTrainer()
  if opts.mode == 'generate' then
    xprintln('This is generate mode!')
    trainer:load(opts.modelPath)
    local inDataPaths = {
      train = opts.inTrain, 
      valid = opts.inValid, 
      test = opts.inTest
    }
    inDataPaths.outvalid = opts.outValid
    inDataPaths.outtest = opts.outTest
    trainer:createTrainData(inDataPaths, opts.outTrainDataPath, opts.language, opts.uDVocab)
    xprintln('Create training data done!')
  elseif opts.mode == 'train' then
    xprintln('This is train mode!')
    trainer:load(opts.modelPath)
    trainer:trainLabeledClassifier(opts)
    xprintln('Training done!')
  else
    error('only support [generate] and [train] mode')
  end
end

main()