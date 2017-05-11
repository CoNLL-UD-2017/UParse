
local MultiModelOpts = {}

function MultiModelOpts.getOpts()
  local cmd = torch.CmdLine()
  cmd:text('====== Select Network ======')
  cmd:option('--seed', 123, 'random seed')
  cmd:option('--model', 'SelectNet', 'model options: SelectNet or SelectNetPos')
  cmd:option('--freqCut', 1, 'for word frequencies')
  cmd:option('--ignoreCase', false, 'whether you will ignore the case')
  cmd:option('--maxNVocab', 0, 'you can also set maximum number of vocabulary')
  cmd:option('--train', '', 'train files directory')
  cmd:option('--valid', '', 'dev files directory')
  cmd:option('--test', '', 'test files directory')
  cmd:option('--dataDir', '/afs/inf.ed.ac.uk/user/s14/s1459234/Projects/dense_parser/data/treebank/multi', 'test files directory')
  cmd:option('--vocabFile', '/afs/inf.ed.ac.uk/group/project/img2txt/dep_parser/dataset/vocab.t7', 'vocabulary file')
  cmd:option('--uDVocab', '/afs/inf.ed.ac.uk/group/project/img2txt/dep_parser/vocab', 'directory of UD vocab')
  cmd:option('--maxEpoch', 30, 'maximum number of epochs')
  cmd:option('--batchSize', 7, '')
  cmd:option('--validBatchSize', 32, '')
  cmd:option('--modelDir', '', 'model directory')
  cmd:option('--feats', 'we,upos,li', 'type: we, upos, xpos, xfeats, lid. For example: we,upos,lid')
  cmd:option('--feat_dims', '50,30,10', 'dimension for each features')
  cmd:option('--nhid', 100, 'hidden unit size')
  cmd:option('--nlayers', 1, 'number of hidden layers')
  cmd:option('--lr', 0.1, 'learning rate')
  cmd:option('--lrDiv', 0, 'learning rate decay when there is no significant improvement. 0 means turn off')
  cmd:option('--minImprovement', 1.001, 'if improvement on log likelihood is smaller then patient --')
  cmd:option('--optimMethod', 'AdaGrad', 'optimization algorithm')
  cmd:option('--modelType', '', 'delex, if using delexicalized parser, otherwise empty')
  cmd:option('--gradClip', 5, '> 0 means to do Pascanu et al.\'s grad norm rescale http://arxiv.org/pdf/1502.04623.pdf; < 0 means to truncate the gradient larger than gradClip; 0 means turn off gradient clip')
  cmd:option('--initRange', 0.1, 'init range')
  cmd:option('--seqLen', 150, 'maximum seqence length')
  cmd:option('--maxTrainLen', 120, 'maximum train sentence length')
  cmd:option('--useGPU', false, 'use GPU')
  cmd:option('--patience', 1, 'stop training if no lower valid PPL is observed in [patience] consecutive epoch(s)')
  cmd:option('--save', 'model.t7', 'save model path')
  
  cmd:option('--disableEearlyStopping', false, 'no early stopping during training')
  
  cmd:text()
  cmd:text('Options for long jobs')
  cmd:option('--savePerEpoch', false, 'save model every epoch')
  cmd:option('--saveBeforeLrDiv', false, 'save model before lr div')
  
  cmd:text()
  cmd:text('Options for regularization')
  cmd:option('--dropout', 0, 'dropout rate (dropping)')
  cmd:text()
  cmd:text('Options for rec dropout')
  cmd:option('--recDropout', 0, 'recurrent dropout')
  
  cmd:text()
  cmd:text('Options for Word Embedding initialization')
  cmd:option('--wordEmbedding', '', 'word embedding path')
  cmd:option('--embedOption', 'init', 'options: init, fineTune (if you use fineTune option, you must specify fineTuneFactor)')
  cmd:option('--fineTuneFactor', 0, '0 mean not upates, other value means such as 0.01')
  
  cmd:text()
  cmd:text('Options for evaluation Standard')
  cmd:option('--evalType', 'stanford', 'stanford or conllx')
  
  local opts = cmd:parse(arg)
  MultiModelOpts.initOpts(opts)
  
  return opts
end

function MultiModelOpts.initOpts(opts)
  -- for different optimization algorithms
  local optimMethods = {'AdaGrad', 'Adam', 'AdaDelta', 'SGD'}
  if not table.contains(optimMethods, opts.optimMethod) then
    error('invalid optimization method! ' .. opts.optimMethod)
  end
  
  opts.curLR = opts.lr
  opts.minLR = 1e-7
  opts.sgdParam = {learningRate = opts.lr}
  if opts.optimMethod == 'AdaDelta' then
    opts.rho = 0.95
    opts.eps = 1e-6
    opts.sgdParam.rho = opts.rho
    opts.sgdParam.eps = opts.eps
  elseif opts.optimMethod == 'SGD' then
    if opts.lrDiv <= 1 then
      opts.lrDiv = 2
    end
  end
  
  torch.manualSeed(opts.seed)
  if opts.useGPU then
    require 'cutorch'
    require 'cunn'
    cutorch.manualSeed(opts.seed)
  end
end

return MultiModelOpts

