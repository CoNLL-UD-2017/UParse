
local DataIter = torch.class('DepPosDataIter')

function DataIter.conllx_iter(infile)
  local fin = io.open(infile)
  local lang_id

  local filename = getFileName(infile)
  -- print(filename)
  if filename:starts('label_train') then
    lang_id = filename:splitc('.')[3]
  else
    lang_id = filename:splitc('-')[2]
  end

  return function()
    local items = {}
    while true do
      local line = fin:read()
      if line == nil then 
        break 
      end
      line = line:trim()
      if line:len() == 0 then
        break
      end
      local fields = line:splitc('\t')
      assert(#fields == 10, 'MUST have 10 fields')
      local item = {
        p1 = tonumber(fields[1]), 
        wd = fields[2] .. '_' .. lang_id, 
        pos = fields[4], 
        p2 = fields[7], 
        rel = fields[8], 
        lid = lang_id}
      table.insert(items, item)
    end
    if #items > 0 then
      return items
    else
      fin:close()
    end
  end
end


function DataIter.createDepRelVocab(infile, uDrel)
  local lbl_freq = {}
  local lbl_vec = {}
  local diter = DataIter.conllx_iter(infile)
  for sent in diter do
    for _, ditem in ipairs(sent) do
      local rel = ditem.rel
      local freq = lbl_freq[rel]
      if freq == nil then
        lbl_vec[#lbl_vec + 1] = rel
        lbl_freq[rel] = 1
      else
        lbl_freq[rel] = freq + 1
      end
    end
  end

  -- add UD rel not available in the training data
  local fin = io.open(uDrel)
  while true do
    local line = fin:read()
    if line == nil then break end
    line = line:trim()
    if line:len() == 0 then break end
    local ud_rel = line
    if lbl_freq[ud_rel] == nil then
      lbl_vec[#lbl_vec + 1] = ud_rel
      lbl_freq[ud_rel] = 1
    end
  end
  
  local rel2idx = {}
  for i, r in ipairs(lbl_vec) do
    rel2idx[r] = i
  end
  
  local lbl_vocab = {}
  lbl_vocab.rel2idx = rel2idx
  lbl_vocab.idx2rel = lbl_vec
  
  return lbl_vocab
end

function DataIter.createDepRelMultiVocab(train_dir, uDrel)
  local lbl_freq = {}
  local lbl_vec = {}
  local diter = {}
  local train_files = listDir(train_dir)

  for _, infile in ipairs(train_files) do
    diter[#diter + 1] = DataIter.conllx_iter(infile)
  end

  for _, d in ipairs(diter) do
    for sent in d do
      for _, ditem in ipairs(sent) do
        local rel = ditem.rel
        local freq = lbl_freq[rel]
        if freq == nil then
          lbl_vec[#lbl_vec + 1] = rel
          lbl_freq[rel] = 1
        else
          lbl_freq[rel] = freq + 1
        end
      end
    end
  end

  -- add UD rel not available in the training data
  local fin = io.open(uDrel)
  while true do
    local line = fin:read()
    if line == nil then break end
    line = line:trim()
    if line:len() == 0 then break end
    local ud_rel = line
    if lbl_freq[ud_rel] == nil then
      lbl_vec[#lbl_vec + 1] = ud_rel
      lbl_freq[ud_rel] = 1
    end
  end
  
  local rel2idx = {}
  for i, r in ipairs(lbl_vec) do
    rel2idx[r] = i
  end
  
  local lbl_vocab = {}
  lbl_vocab.rel2idx = rel2idx
  lbl_vocab.idx2rel = lbl_vec
  
  return lbl_vocab
end

function DataIter.getDataSize(infiles)
  local sizes = {}
  for _, infile in ipairs(infiles) do
    local size = 0
    local diter = DataIter.conllx_iter(infile)
    for ds in diter do
      size = size + 1
    end
    sizes[#sizes + 1] = size
  end
  
  return sizes
end

function DataIter.showVocab(vocab)
  for k, v in pairs(vocab) do
    xprint(k)
    if type(v) == 'table' then
      print ' -- table' 
    else
      print( ' -- ' .. tostring(v) )
    end
  end
end

function DataIter.createMultiVocab(train_dir, ignoreCase, freqCut, maxNVocab, uDpos)
  local wordFreq = {}
  local wordVec = {}
  local diter = {}
  local train_files = listDir(train_dir)

  for _, infile in ipairs(train_files) do
    diter[#diter + 1] = DataIter.conllx_iter(infile)
  end
  
  -- add words into vocabulary
  local function addwd(wd)
    local wd = ignoreCase and wd:lower() or wd
    local freq = wordFreq[wd]
    if freq == nil then
      wordFreq[wd] = 1
      wordVec[#wordVec + 1] = wd
    else
      wordFreq[wd] = freq + 1
    end
  end
  
  -- add POS tag to vocabulary
  local tagFreq = {}
  local idx2pos = {}
  local function addtag(tag)
    local freq = tagFreq[tag]
    if freq == nil then
      tagFreq[tag] = 1
      idx2pos[#idx2pos + 1] = tag
    else
      tagFreq[tag] = freq + 1
    end
  end
  
  for _, d in ipairs(diter) do
    for sent in d do
      addwd('###root###')
      addtag('###root###')
      for _, ditem in ipairs(sent) do
        addwd(ditem.wd)
        addtag(ditem.pos)
      end
    end
  end

  -- add UD pos not available in the training data
  local fin = io.open(uDpos)
  while true do
    local line = fin:read()
    if line == nil then break end
    line = line:trim()
    if line:len() == 0 then break end
    addtag(line)
  end
  
  local pos2idx = {}
  for i, pos in pairs(idx2pos) do
    pos2idx[pos] = i
  end

  printf('total number of tags: %d\n', #idx2pos)
  
  local idx2word
  if freqCut and freqCut >= 0 then
    idx2word = {}
    idx2word[#idx2word + 1] = 'UNK'
    for _, wd in ipairs(wordVec) do
      if wordFreq[wd] > freqCut then idx2word[#idx2word + 1] = wd end
    end
    
    printf('original word count = %d, after freq cut = %d, word count = %d\n', #wordVec, freqCut, #idx2word)
  end
  
  if maxNVocab and maxNVocab > 0 then
    if #idx2word > 0 then
      print( 'WARNING: rewrote idx2word with maxNVocab = ' .. maxNVocab )
    end
    idx2word = {}
    idx2word[#idx2word + 1] = 'UNK'
    local wfs = {}
    for _, k in ipairs(wordVec) do table.insert(wfs, {k, wordFreq[k]}) end
    table.sort(wfs, function(x, y) return x[2] > y[2] end)
    local lfreq = -1
    for cnt, kv in ipairs(wfs) do
      idx2word[#idx2word + 1] = kv[1]
      lfreq = kv[2]
      if cnt >= maxNVocab-1 then break end
    end
    printf('original word count = %d, after maxNVocab = %d, word count = %d, lowest freq = %d\n', #wordVec, maxNVocab, #idx2word, lfreq)
  end
  
  local word2idx = {}
  for i, w in ipairs(idx2word) do word2idx[w] = i end
  local vocab = {word2idx = word2idx, idx2word = idx2word,
    freqCut = freqCut, ignoreCase = ignoreCase, maxNVocab = maxNVocab,
    UNK_STR = 'UNK', UNK = word2idx['UNK'],
    ROOT_STR = '###root###', ROOT = word2idx['###root###']}
  vocab['nvocab'] = table.len(word2idx)
  
  vocab['idx2pos'] = idx2pos
  vocab['pos2idx'] = pos2idx
  vocab['npos'] = table.len(pos2idx)
  vocab['ROOT_POS'] = pos2idx['###root###']
  vocab['ROOT_POS_STR'] = '###root###'
  
  DataIter.showVocab(vocab)
  
  return vocab
end

function DataIter.createVocab(infile, ignoreCase, freqCut, maxNVocab, uDpos)
  local wordFreq = {}
  local wordVec = {}
  local diter = DataIter.conllx_iter(infile)
  
  local function addwd(wd)
    local wd = ignoreCase and wd:lower() or wd
    local freq = wordFreq[wd]
    if freq == nil then
      wordFreq[wd] = 1
      wordVec[#wordVec + 1] = wd
    else
      wordFreq[wd] = freq + 1
    end
  end
  
  local tagFreq = {}
  local idx2pos = {}
  local function addtag(tag)
    local freq = tagFreq[tag]
    if freq == nil then
      tagFreq[tag] = 1
      idx2pos[#idx2pos + 1] = tag
    else
      tagFreq[tag] = freq + 1
    end
  end
  
  for sent in diter do
    addwd('###root###')
    addtag('###root###')
    for _, ditem in ipairs(sent) do
      addwd(ditem.wd)
      addtag(ditem.pos)
    end
  end

  -- add UD pos not available in the training data
  local fin = io.open(uDpos)
  while true do
    local line = fin:read()
    if line == nil then break end
    line = line:trim()
    if line:len() == 0 then break end
    addtag(line)
  end
  
  local pos2idx = {}
  for i, pos in pairs(idx2pos) do
    pos2idx[pos] = i
  end

  printf('total number of tags: %d\n', #idx2pos)
  
  local idx2word
  if freqCut and freqCut >= 0 then
    idx2word = {}
    idx2word[#idx2word + 1] = 'UNK'
    for _, wd in ipairs(wordVec) do
      if wordFreq[wd] > freqCut then idx2word[#idx2word + 1] = wd end
    end
    
    printf('original word count = %d, after freq cut = %d, word count = %d\n', #wordVec, freqCut, #idx2word)
  end
  
  if maxNVocab and maxNVocab > 0 then
    if #idx2word > 0 then
      print( 'WARNING: rewrote idx2word with maxNVocab = ' .. maxNVocab )
    end
    idx2word = {}
    idx2word[#idx2word + 1] = 'UNK'
    local wfs = {}
    for _, k in ipairs(wordVec) do table.insert(wfs, {k, wordFreq[k]}) end
    table.sort(wfs, function(x, y) return x[2] > y[2] end)
    local lfreq = -1
    for cnt,  kv in ipairs(wfs) do
      idx2word[#idx2word + 1] = kv[1]
      lfreq = kv[2]
      if cnt >= maxNVocab-1 then break end
    end
    printf('original word count = %d, after maxNVocab = %d, word count = %d, lowest freq = %d\n', #wordVec, maxNVocab, #idx2word, lfreq)
  end
  
  local word2idx = {}
  for i, w in ipairs(idx2word) do word2idx[w] = i end
  local vocab = {word2idx = word2idx, idx2word = idx2word,
    freqCut = freqCut, ignoreCase = ignoreCase, maxNVocab = maxNVocab,
    UNK_STR = 'UNK', UNK = word2idx['UNK'],
    ROOT_STR = '###root###', ROOT = word2idx['###root###']}
  vocab['nvocab'] = table.len(word2idx)
  
  vocab['idx2pos'] = idx2pos
  vocab['pos2idx'] = pos2idx
  vocab['npos'] = table.len(pos2idx)
  vocab['ROOT_POS'] = pos2idx['###root###']
  vocab['ROOT_POS_STR'] = '###root###'
  
  DataIter.showVocab(vocab)
  
  return vocab
end


function DataIter.toBatch(sents, vocab, batchSize)
  local dtype = 'torch.LongTensor'
  local maxn = -1
  for _, sent in ipairs(sents) do if sent:size(1) > maxn then maxn = sent:size(1) end end

  assert(maxn ~= -1)
  local x = (torch.ones(maxn + 1, batchSize) * vocab.UNK):type(dtype)
  local x_mask = torch.zeros(maxn + 1, batchSize)
  local x_pos = (torch.ones(maxn + 1, batchSize) * vocab.ROOT_POS):type(dtype)
  local y = torch.zeros(maxn, batchSize):type(dtype)

  x[{ 1, {} }] = vocab.ROOT
  x_pos[{ 1, {} }] = vocab.ROOT_POS
  x_mask[{ 1, {} }] = 1
  for i, sent in ipairs(sents) do
    local slen = sent:size(1)
    x[{ {2, slen + 1}, i }] = sent[{ {}, 1 }]
    x_pos[{ {2, slen + 1}, i }] = sent[{ {}, 3 }]
    x_mask[{ {2, slen + 1}, i }] = 1
    y[{ {1, slen}, i }] = sent[{ {}, 2 }]
  end
  
  return x, x_mask, x_pos, y
end

function DataIter.sent2dep(vocab, sent)
  local d = {}
  local word2idx = vocab.word2idx
  local pos2idx = vocab.pos2idx
  for _, ditem in ipairs(sent) do
    local wd = vocab.ignoreCase and ditem.wd:lower() or ditem.wd
    local wid = word2idx[wd] or vocab.UNK
    local posid = pos2idx[ditem.pos]
    d[#d + 1] = {wid, ditem.p2 + 1, posid}
  end
  return torch.Tensor(d), #d
end

function DataIter.multisent2dep(vocab, sent)
  local d = {}
  local word2idx = vocab.word2idx
  local pos2idx = vocab.pos2idx
  local lang_id = ''
  for _, ditem in ipairs(sent) do
    local wd = vocab.ignoreCase and ditem.wd:lower() or ditem.wd
    local wid = word2idx[wd] or vocab.UNK
    local posid = pos2idx[ditem.pos]
    lang_id = ditem.lid
    d[#d + 1] = {wid, ditem.p2 + 1, posid}
  end
  return torch.Tensor(d), #d, lang_id
end

function DataIter.createBatch(vocab, infile, batchSize, maxlen)
  maxlen = maxlen or 100
  local diter = DataIter.conllx_iter(infile)
  local isEnd = false
  
  return function()
    if not isEnd then
      local sents = {}
      for i = 1, batchSize do
        local sent = diter()
        if sent == nil then isEnd = true break end
        local s, len = DataIter.sent2dep(vocab, sent)
        if len <= maxlen then 
          sents[#sents + 1] = s
        else
          print ( 'delete sentence with length ' .. tostring(len) )
        end
      end
      if #sents > 0 then
        return DataIter.toBatch(sents, vocab, batchSize)
      end
      
    end
  end
end

function DataIter.createBatchLabel(vocab, rel_vocab, infile, batchSize, maxlen)
  maxlen = maxlen or 100
  local diter = DataIter.conllx_iter(infile)
  local isEnd = false
  local rel2idx = rel_vocab.rel2idx
  
  return function()
    if not isEnd then
      local sents = {}
      local sent_rels = {}
      local sent_ori_rels = {}
      for i = 1, batchSize do
        local sent = diter()
        if sent == nil then isEnd = true break end
        local s, len = DataIter.sent2dep(vocab, sent)
        if len <= maxlen then 
          sents[#sents + 1] = s
          local sent_rel = {}
          local sent_ori_rel = {}
          for i, item in ipairs(sent) do
            sent_rel[i] = rel2idx[item.rel]
            sent_ori_rel[i] = item.rel
          end
          sent_rels[#sent_rels + 1] = sent_rel
          sent_ori_rels[#sent_ori_rels + 1] = sent_ori_rel
        else
          print ( 'delete sentence with length ' .. tostring(len) )
        end
      end

      if #sents > 0 then
        local x, x_mask, x_pos, y = DataIter.toBatch(sents, vocab, batchSize)
        return x, x_mask, x_pos, y, sent_rels, sent_ori_rels
      end
    end
  end
end

function DataIter.createBatchSort(vocab, infile, batchSize, maxlen)
  maxlen = maxlen or 100
  local diter = DataIter.conllx_iter(infile)
  local all_sents = {}
  for sent in diter do
    local s, len = DataIter.sent2dep(vocab, sent)
    all_sents[#all_sents + 1] = s
  end
  -- print(all_sents[1])
  table.sort(all_sents, function(a, b)  return a:size(1) < b:size(1) end)
  
  local cnt = 0
  local ndata = #all_sents
  
  return function()
    
    local sents = {}
    for i = 1, batchSize do
      cnt = cnt + 1
      if cnt <= ndata then
        sents[#sents + 1] = all_sents[cnt]
      end
    end
    
    if #sents > 0 then
      return DataIter.toBatch(sents, vocab, batchSize)
    end
    
  end
end

function DataIter.loadAllSents(vocab, infile, maxlen)
  local diter = DataIter.conllx_iter(infile)
  local all_sents = {}
  local del_cnt = 0
  for sent in diter do
    local s, len = DataIter.sent2dep(vocab, sent)
    if len <= maxlen then
      all_sents[#all_sents + 1] = s
    else
      del_cnt = del_cnt + 1
    end
  end
  if del_cnt > 0 then
    printf( 'WARNING!!! delete %d sentences that longer than %d\n', del_cnt, maxlen)
  end
  return all_sents
end

function DataIter.loadMultiSents(vocab, train_dir, maxlen, label)
  local train_files = listDir(train_dir)
  local sents_per_lang = 1000000
  local diter = {}
  local multi_sents = {}

  for _, f in ipairs(train_files) do
    diter[#diter + 1] = DataIter.conllx_iter(f)
  end

  for _, d in ipairs(diter) do
    local all_sents = {}
    local lang_id = ''
    local del_cnt = 0
    for sent in d do
      local s, len
      s, len, lang_id = DataIter.multisent2dep(vocab, sent)
      if len <= maxlen then
        if not label then
          all_sents[#all_sents + 1] = s
        else
          local s_rel = {}
          local sent_rel = {}
          for i, item in ipairs(sent) do
            sent_rel[#sent_rel + 1] = item.rel
          end
          s_rel.sentdep = s
          s_rel.sentrel = sent_rel
          all_sents[#all_sents + 1] = s_rel
        end
      else
        del_cnt = del_cnt + 1
      end
    end
    if del_cnt > 0 then
      printf( 'WARNING!!! delete %d sentences that longer than %d\n', del_cnt, maxlen)
    end
    xprintln('Language: %s, total sents: %d', lang_id, #all_sents)
    if #all_sents < sents_per_lang then
      sents_per_lang = #all_sents
    end
    multi_sents[lang_id] = all_sents
  end
  local total_sents = sents_per_lang * #train_files
  xprintln('Total final training sentences: %d sents.', total_sents)
  return multi_sents
end

function DataIter.createBatchShuffleSort(all_sents_, vocab, batchSize, sort_flag, shuffle)
  assert(sort_flag ~= nil and (shuffle == true or shuffle == false))
  
  local function shuffle_dataset(all_sents)
    local tmp_sents = {}
    local idxs = torch.randperm(#all_sents)
    for i = 1, idxs:size(1) do
      tmp_sents[#tmp_sents + 1] = all_sents[ idxs[i] ]
    end
    return tmp_sents
  end
  
  local all_sents
  if shuffle then
    all_sents = shuffle_dataset(all_sents_)
  end
  
  local len_idxs = {}
  for i, sent in ipairs(all_sents) do
    len_idxs[#len_idxs + 1] = {sent:size(1), i}
  end
  
  local kbatches = sort_flag * batchSize
  local new_idxs = {}
  local N = #len_idxs
  for istart = 1, N, kbatches do
    iend = math.min(istart + kbatches - 1, N)
    local tmpa = {}

    for i = istart, iend do
      tmpa[#tmpa + 1] = len_idxs[i]
    end

    -- sort sentences based on length
    table.sort(tmpa, function( a, b ) return a[1] < b[1] end)
    
    for _, tmp in ipairs(tmpa) do
      new_idxs[#new_idxs + 1] = tmp[2]
    end
  end
  
  local final_all_sents = {}
  for _, idx in ipairs(new_idxs) do
    final_all_sents[#final_all_sents + 1] = all_sents[idx]
  end
  
  local cnt, ndata = 0, #final_all_sents
  return function()
    local sents = {}
    for i = 1, batchSize do
      cnt = cnt + 1
      print(cnt, ndata)
      if cnt > ndata then break end
      sents[#sents + 1] = final_all_sents[cnt]
    end
    
    if #sents > 0 then
      return DataIter.toBatch(sents, vocab, batchSize)
    end
    
  end
end

function DataIter.createBatchMultiShuffleSort(all_sents_, vocab, batchSize, sort_flag, shuffle)
  assert(sort_flag ~= nil and (shuffle == true or shuffle == false))
  
  local function shuffle_dataset(all_sents)
    local tmp_sents = {}
    local idxs = torch.randperm(#all_sents)
    for i = 1, idxs:size(1) do
      tmp_sents[#tmp_sents + 1] = all_sents[ idxs[i] ]
    end
    return tmp_sents
  end
  
  local all_sents = {}
  local sents_per_lang = 1000000
  local smallest_treebank = ''
  if shuffle then
    for lang_id, lang_sents in pairs(all_sents_) do
      all_sents[lang_id] = shuffle_dataset(lang_sents)
      -- count the number of sentences in the smallest treebank
      if #lang_sents < sents_per_lang then
        sents_per_lang = #lang_sents
        smallest_treebank = lang_id
      end
    end
  end

  -- sents_per_lang = 140
  local final_all_sents = {}
  for i = 1, sents_per_lang do
    for lang_id, lang_sents in pairs(all_sents) do
      final_all_sents[#final_all_sents + 1] = lang_sents[i]
    end
  end
  
  local cnt, ndata = 0, #final_all_sents
  return function()
    local sents = {}
    for i = 1, batchSize do
      cnt = cnt + 1
      if cnt > ndata then break end
      sents[#sents + 1] = final_all_sents[cnt]
    end
    if #sents > 0 then
      return DataIter.toBatch(sents, vocab, batchSize)
    end
  end
end

function DataIter.createBatchLabelMulti(vocab, rel_vocab, train_dir, batchSize, maxlen)

  local function shuffle_dataset(all_sents)
    local tmp_sents = {}
    local idxs = torch.randperm(#all_sents)
    for i = 1, idxs:size(1) do
      tmp_sents[#tmp_sents + 1] = all_sents[ idxs[i] ]
    end
    return tmp_sents
  end

  maxlen = maxlen or 100
  multi_sents = DataIter.loadMultiSents(vocab, train_dir, maxlen, true)

  local isEnd = false
  local rel2idx = rel_vocab.rel2idx
  local all_sents = {}
  local sents_per_lang = 1000000
  local smallest_treebank = ''

  for lang_id, lang_sents in pairs(multi_sents) do
    all_sents[lang_id] = shuffle_dataset(lang_sents)
    -- count the number of sentences in the smallest treebank
    if #lang_sents < sents_per_lang then
      sents_per_lang = #lang_sents
      smallest_treebank = lang_id
    end
  end

  -- sents_per_lang = 140
  local final_all_sents = {}
  for i = 1, sents_per_lang do
    for lang_id, lang_sents in pairs(all_sents) do
      final_all_sents[#final_all_sents + 1] = lang_sents[i]
    end
  end
  
  local cnt = 0
  local ndata = #final_all_sents
  
  return function()
    local sents = {}
    local sent_rels = {}
    local sent_ori_rels = {}
    
    for i = 1, batchSize do
      cnt = cnt + 1
      if cnt > ndata then break end
      local s_rel = final_all_sents[cnt]
      sents[#sents + 1] = s_rel.sentdep
      local sent_rel = {}
      local sent_ori_rel = {}
      for _, rel in ipairs(s_rel.sentrel) do
        sent_rel[#sent_rel + 1] = rel2idx[rel]
        sent_ori_rel[#sent_ori_rel + 1] = rel
      end
      sent_rels[#sent_rels + 1] = sent_rel
      sent_ori_rels[#sent_ori_rels + 1] = sent_ori_rel
    end

    if #sents > 0 then
      local x, x_mask, x_pos, y = DataIter.toBatch(sents, vocab, batchSize)
      return x, x_mask, x_pos, y, sent_rels, sent_ori_rels
    end
  end
end

local function main()  
  require '../utils/shortcut'
  local infile = '/afs/inf.ed.ac.uk/user/s14/s1459234/Projects/dense_parser/data/treebank/UD_Italian/cleaned-it-ud-train.conllu'
  local vocab = DepPosDataIter.createVocab(infile, true, 1, 0, '/afs/inf.ed.ac.uk/user/s14/s1459234/Projects/dense_parser/vocab/ud_pos.vocab')
  print 'get vocab done!'
  
  local validfile = '/afs/inf.ed.ac.uk/user/s14/s1459234/Projects/dense_parser/data/treebank/UD_Italian/cleaned-it-ud-dev.conllu'
  local batchIter = DepPosDataIter.createBatch(vocab, validfile, 32, 100)
  local cnt = 0
  for x, x_mask, x_pos, y in batchIter do
    cnt = cnt + 1
    if cnt < 3 then
      print 'x = '
      print(x)
      print 'x_mask = '
      print(x_mask)
      print 'x_pos = '
      print(x_pos)
      print 'y = '
      print(y)
    end
  end
  print( 'total ' .. cnt )
end

if not package.loaded['DepPosDataIter'] then
  main()
end
