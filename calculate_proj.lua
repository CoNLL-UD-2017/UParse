
require '.'
require 'shortcut'
require 'SelectNetPos'
require 'DepPosDataIter'
require 'PostDepGraph'

local CP = torch.class('ProjectiveTree')


function CP:calc(trainFile)
  local dep_iter = DepPosDataIter.conllx_iter(trainFile)
  local sent_idx = 0
  local connected_count = 0
  for dsent in dep_iter do
  	sent_idx = sent_idx + 1
    local dgraph = PostDepGraph(dsent)
    if (dgraph:checkConnectivity() and dgraph:isProjective()) then
      connected_count = connected_count + 1
    end
  end
  printf('%d/%d = %f are projective trees\n', connected_count, sent_idx, connected_count/sent_idx)
end

local function getOpts()
  local cmd = torch.CmdLine()
  cmd:option('--train', '/afs/inf.ed.ac.uk/user/s14/s1459234/Projects/dense_parser/data/treebank/UD_Italian/cleaned-it-ud-train.conllu', 'train data')  
  return cmd:parse(arg)
end

local function main()
  local opts = getOpts()
  local dense = ProjectiveTree()
  print(opts.train)
  dense:calc(opts.train)
end


if not package.loaded['calculate_proj'] then
  main()
end

