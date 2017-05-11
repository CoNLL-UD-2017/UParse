
include '../utils/shortcut.lua'


local function loadVocab(vocabF)
  local vocab = {}
  local fin = io.open(vocabF)
  while true do
    local line = fin:read()
    if line == nil then break end
    vocab[line] = 1
  end
  return vocab
end

local function extractTxtWE(vocabF, WEPath, WEOutPath)
  local vocab = loadVocab(vocabF)
  
  local fin = io.open(WEPath)
  local msize, nsize = 0, 0
  local cnt = 0
  local wetable = {}
  local idx2word = {}
  local word_dict = {}

  local vocabSize = 0
  local embDim = 0

  while true do
    local line = fin:read()
    if line == nil then break end
    local fields = line:splitc(' ')
    -- check different format: word2vec vs. glove format
    if #fields == 2 then
      local fields = line:splitc(' ')
      vocabSize = fields[1]
      embDim = fields[2]
      print('Embedding dimension ' .. embDim)
      readFirst = false
    else
      local width = #fields - 1
      if nsize == 0 then
        nsize = width
      else
        assert(nsize == width)
      end
      local word = fields[1]
      if vocab[word] ~= nil then
        -- print('match')
        local update = false
        if word_dict[word] == nil then
          msize = msize + 1
          idx2word[msize] = word
        else
          update = true
        end

        local v = {}
        for i = 2, width + 1 do
          table.insert(v, tonumber(fields[i]))
        end
        if update == false then
          table.insert(wetable, v)
          word_dict[word] = #wetable
          assert(#wetable == msize)
        end
      end

      cnt = cnt + 1
      if cnt % 1000 == 0 then
        printf('cnt = %d\n', cnt)
      end
    end
  end
  print('totally ' .. msize .. ' lines remain')
  
  local word2idx = {}
  for i, w in pairs(idx2word) do
    word2idx[w] = i
  end
  print(#word2idx, #idx2word, #wetable)
  
  local final_we = torch.FloatTensor(wetable)
  print 'begin to save'
  torch.save(WEOutPath, {final_we, word2idx, idx2word})
  print( 'save done at ' .. WEOutPath)
end

local function main()
  local cmd = torch.CmdLine()
  cmd:option('--vocab', 'ptb.train.tmp.vocab.t7', 'path for vocab file')
  cmd:option('--wepath', 'glove.840B.300d.txt', 'glove vectors')
  cmd:option('--weoutpath', 'ptb.glove.840B.300d.t7', 'output embedding file')
  local opts = cmd:parse(arg)
  
  extractTxtWE(opts.vocab, opts.wepath, opts.weoutpath)
end

main()

