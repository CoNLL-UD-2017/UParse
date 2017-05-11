
include '../utils/shortcut.lua'

local function extractTxtWE(WEPath, WEOutPath)
  local fin = io.open(WEPath)
  local msize, nsize = 0, 0
  local cnt = 0
  local wetable = {}
  local idx2word = {}
  local word_dict = {}
  local overlap_words = {}
  local vocabSize = 0
  local embDim = 0

  local run = true
  while run do
    local line = fin:read()
    if line == nil then break end
    local fields = line:splitc(' ')
    if #fields == 2 then
      vocabSize = fields[1]
      embDim = fields[2]
      print('Embedding dimension ' .. embDim)
    else
      local width = #fields - 1
      if nsize == 0 then
        nsize = width
      else
        assert(nsize == width)
      end
      local word = fields[1]
      local update = false
      if word_dict[word] ~= nil then
        update = true
      else
        msize = msize + 1
        idx2word[msize] = word
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

      cnt = cnt + 1
      if cnt % 10000 == 0 then
        printf('cnt = %d\n', cnt)
      end
      if cnt > 140000 then
        run = false
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
  cmd:option('--lang', '', 'bridge language')
  cmd:option('--wepath', '', 'fastText word vectors')
  cmd:option('--weoutpath', '', 'output embedding files dir')
  local opts = cmd:parse(arg)
  
  extractTxtWE(opts.wepath, opts.weoutpath)
end

main()

