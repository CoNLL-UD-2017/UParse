
require 'basic'
require 'PReplicate'
require 'Linear3D'
require 'Contiguous'
require 'EMaskedClassNLLCriterion'
require 'LookupTable_ft'

local model_utils = require 'model_utils'

local SelectNet = torch.class('SelectNetRich', 'BModel')

function SelectNet:__init(opts)
  self.opts = opts
  self.name = 'SelectNetRich'
  opts.nvocab = opts.vocab.nvocab
  self:createNetwork(opts)
  
  if opts.optimMethod == 'AdaGrad' then
    self.optimMethod = optim.adagrad
  elseif opts.optimMethod == 'Adam' then
    self.optimMethod = optim.adam
  elseif opts.optimMethod == 'AdaDelta' then
    self.optimMethod = optim.adadelta
  elseif opts.optimMethod == 'SGD' then
    self.optimMethod = optim.sgd
  end
end

function SelectNet:transData(d, cpu_type)
  if self.opts.useGPU then 
    return d:cuda() 
  else
    if cpu_type then
      return d:type(cpu_type)
    else
      return d
    end
  end
end

function SelectNet:createLSTM(x_t, c_tm1, h_tm1, nin, nhid, label)
  -- compute activations of four gates all together
  local x2h = nn.Linear(nin, nhid * 4)(x_t)
  local h2h = nn.Linear(nhid, nhid * 4)(h_tm1)
  local allGatesActs = nn.CAddTable()({x2h, h2h})
  local allGatesActsSplits = nn.SplitTable(2)( nn.Reshape(4, nhid)(allGatesActs) )

  -- unpack all gate activations
  local i_t = nn.Sigmoid()( nn.SelectTable(1)( allGatesActsSplits ) )
  local f_t = nn.Sigmoid()( nn.SelectTable(2)( allGatesActsSplits ) )
  local o_t = nn.Sigmoid()( nn.SelectTable(3)( allGatesActsSplits ) )
  local n_t = nn.Tanh()( nn.SelectTable(4)( allGatesActsSplits ) )
  
  if self.opts.recDropout > 0 then
    n_t = nn.Dropout(self.opts.recDropout)(n_t)
    printf( 'lstm [%s], RECURRENT dropout = %f\n', label, self.opts.recDropout) 
  end
  
  -- compute new cell
  local c_t = nn.CAddTable()({
      nn.CMulTable()({ i_t, n_t }),
      nn.CMulTable()({ f_t, c_tm1 })
    })
  
  -- compute new hidden state
  local h_t = nn.CMulTable()({ o_t, nn.Tanh()( c_t ) })
  return c_t, h_t
end

function SelectNet:createDeepLSTM(opts, label)
  local feats = {}
  local feat_dims = {}
  local features = {}
  local feat_emb = {}
  local input_dim = 0

  local we_idx = 0
  local we_dim = 0

  for _, token in ipairs(opts.feats:splitc(',')) do
    feats[#feats + 1] = token 
    if token == 'we' then
      we_idx = #feats
    end
  end
  for _, token in ipairs(opts.feat_dims:splitc(',')) do
    feat_dims[#feat_dims + 1] = token
    input_dim = input_dim + token
    if we_idx == #feat_dims then
      we_dim = token
    end
  end
  
  assert(#feats == #feat_dims, 'Number of features and dims should be the same')
  for i = 1, #feats do
    if feats[i] ~= 'we' then
      local ntag = opts.vocab['n' .. feats[i]]
      feat_emb[feats[i]] = nn.LookupTable(ntag, feat_dims[i])
    end
  end

  -- word lookup table
  local emb = (opts.embedOption ~= nil and opts.embedOption == 'fineTune')
    and LookupTable_ft(opts.nvocab, we_dim)
    or nn.LookupTable(opts.nvocab, we_dim)
  
  local init_param = {}
  local x_t = nn.Identity()()
  local s_tm1 = nn.Identity()()
  local inputs = {}
  local we = emb(x_t):annotate{name= label ..'lookup'}

  table.insert(init_param, x_t)
  table.insert(inputs, we)
  
  local feat_vec = {}
  local x_feat_t = {}
  for i, feat in ipairs(feats) do
    if feat ~= 'we' then
      x_feat_t[feat] = nn.Identity()()
      feat_vec[feat] = feat_emb[feat](x_feat_t[feat]):annotate{name= label .. feat .. '_lookup'}
      table.insert(init_param, x_feat_t[feat])
      table.insert(inputs, feat_vec[feat])
    end
  end
  table.insert(init_param, s_tm1)

  local in_t = { [0] = nn.JoinTable(2)(inputs) }
  local s_t = {}
  local splits_tm1 = {s_tm1:split(2 * opts.nlayers)}
  
  for i = 1, opts.nlayers do
    local c_tm1_i = splits_tm1[i + i - 1]
    local h_tm1_i = splits_tm1[i + i]
    local x_t_i = in_t[i - 1]
    local c_t_i, h_t_i = nil, nil
    
    if opts.dropout > 0 then
      printf( '%s lstm layer %d, dropout = %f\n', label, i, opts.dropout) 
      x_t_i = nn.Dropout(opts.dropout)(x_t_i)
    end
    
    if i == 1 then
      c_t_i, h_t_i = self:createLSTM(x_t_i, c_tm1_i, h_tm1_i, input_dim, opts.nhid, label .. i)
    else
      c_t_i, h_t_i = self:createLSTM(x_t_i, c_tm1_i, h_tm1_i, opts.nhid, opts.nhid, label .. i)
    end
    s_t[i+i-1] = c_t_i
    s_t[i+i] = h_t_i
    in_t[i] = h_t_i
  end
  
  local model = nn.gModule(init_param, {nn.Identity()(s_t)})
  return self:transData(model)
end

function SelectNet:createAttention(opts)
  -- note you used both forward and backward model
  local nhid = opts.nhid * 2
  -- enc_hs shape: (bs, seqlen, nhid)
  local enc_hs = nn.Identity()()
  local hs = nn.Identity()()
  local seqlen = nn.Identity()()
  local mask = nn.Identity()()
  local mask_sub = nn.Identity()()
  
  local h1 = Linear3D(nhid, nhid)(enc_hs)
	local h2_ = nn.Linear(nhid, nhid)(hs)
	-- local h2 = Contiguous()( PReplicate(2){h2_, seqlen} )
  local h2 = ( PReplicate(2){h2_, seqlen} )
	local h = nn.Tanh()( nn.CAddTable(){h1, h2} )
	local aout = nn.Sum(3)( Linear3D(nhid, 1)(h) )
  
  aout = nn.CAddTable()({ 
      nn.CMulTable()({ aout, mask }),
      mask_sub
    })
  
  local y_prob = nn.LogSoftMax()(aout)
  
  local model = nn.gModule({enc_hs, hs, seqlen, mask, mask_sub}, 
    {y_prob})
  
  return self:transData(model)
end

function SelectNet:createNetwork(opts)
  self.forward_lstm_master = self:createDeepLSTM(opts, 'forward_')
  self.backward_lstm_master = self:createDeepLSTM(opts, 'backward_')
  self:print('create forward and backward LSTM done!')
  self.attention_master = self:createAttention(opts)
  self:print('create attention model done!')

  -- backward_lookup is ignored
  self.params, self.grads = model_utils.combine_selectnet_pos_parameters(self.forward_lstm_master, 
    self.backward_lstm_master, self.attention_master)
  self.params:uniform(-opts.initRange, opts.initRange)
  self:print('#params ' .. self.params:nElement())
  
  -- share forward and backward lookupTable
  self.mod_map = BModel.get_module_map({self.forward_lstm_master, self.backward_lstm_master, self.attention_master})
  self.mod_map.backward_lookup.weight:set( self.mod_map.forward_lookup.weight )
  self.mod_map.backward_lookup.gradWeight:set( self.mod_map.forward_lookup.gradWeight )

  for _, feat in ipairs(opts.feats:splitc(',')) do
    if feat ~= 'we' then
      local f_label = 'forward_' .. feat .. '_lookup'
      local b_label = 'backward_' .. feat .. '_lookup'
      self.mod_map[b_label].weight:set( self.mod_map[f_label].weight )
      self.mod_map[b_label].gradWeight:set( self.mod_map[f_label].gradWeight )
    end
  end
  collectgarbage()
  self:print('forward lstm and backward lstm share parameters')
  
  -- intialize with pre-trained word embedding
  if self.opts.wordEmbedding ~= nil and self.opts.wordEmbedding ~= ''  then
    local net_lookup = self.mod_map.forward_lookup
    self.net_lookup = net_lookup
    if self.opts.embedOption == 'init' then
      model_utils.load_embedding_init(net_lookup, self.opts.vocab, self.opts.wordEmbedding)
    elseif self.opts.embedOption == 'fineTune' then
      model_utils.load_embedding_fine_tune(net_lookup, self.opts.vocab, self.opts.wordEmbedding, self.opts.fineTuneFactor)
    else
      error('invalid option -- ' .. self.opts.embedOption)
    end
  end
  
  if self.opts.embedOption == 'fineTune' then
    -- this will not copy updateMask
    self.forward_lstms = model_utils.clone_many_times_emb_ft(self.forward_lstm_master, opts.seqLen)
    self.backward_lstms = model_utils.clone_many_times_emb_ft(self.backward_lstm_master, opts.seqLen)
  else
    self.forward_lstms = model_utils.clone_many_times(self.forward_lstm_master, opts.seqLen)
    self.backward_lstms = model_utils.clone_many_times(self.backward_lstm_master, opts.seqLen)
  end
  self.attentions = model_utils.clone_many_times(self.attention_master, opts.seqLen)
  self:print('clone model done!')
  
  -- time for dealing with criterions
  self.criterions = {}
  for i = 1, opts.seqLen do
    self.criterions[i] = self:transData(EMaskedClassNLLCriterion())
  end
  
  -- init model
  self:initModel(opts)
end

function SelectNet:initModel(opts)
  self.fwd_h0 = {}
  self.df_fwd_h = {}
  for i = 1, 2 * opts.nlayers do
    self.fwd_h0[i] = self:transData( torch.zeros(opts.batchSize, opts.nhid) )
    self.df_fwd_h[i] = self:transData( torch.zeros(opts.batchSize, opts.nhid) )
  end
  -- print(self.fwd_h0)
  self.fwd_hs = {}
  for i = 0, opts.seqLen do
    local tmp = {}
    for j = 1, 2 * opts.nlayers do
      tmp[j] = self:transData( torch.zeros(opts.batchSize, opts.nhid) )
    end
    self.fwd_hs[i] = tmp
  end
  
  self.bak_h0 = {}
  self.df_bak_h = {}
  for i = 1, 2 * opts.nlayers do
    self.bak_h0[i] = self:transData( torch.zeros(opts.batchSize, opts.nhid) )
    self.df_bak_h[i] = self:transData( torch.zeros(opts.batchSize, opts.nhid) )
  end
  -- print(self.bak_h0)
  self.bak_hs = {}
  for i = 1, opts.seqLen + 1 do
    local tmp = {}
    for j = 1, 2 * opts.nlayers do
      tmp[j] = self:transData( torch.zeros(opts.batchSize, opts.nhid) )
    end
    self.bak_hs[i] = tmp
  end
  
  -- this is for attention model
  self.all_fwd_bak_hs = self:transData( torch.zeros(opts.batchSize * opts.seqLen, 2 * opts.nhid) )
  self.df_all_fwd_bak_hs = self:transData( torch.zeros(opts.batchSize * opts.seqLen, 2 * opts.nhid) )
end

function SelectNet:training()
  self.forward_lstm_master:training()
  self.backward_lstm_master:training()
  self.attention_master:training()
  
  for i = 1, self.opts.seqLen do
    self.forward_lstms[i]:training()
    self.backward_lstms[i]:training()
    self.attentions[i]:training()
  end
end

function SelectNet:evaluate()
  self.forward_lstm_master:evaluate()
  self.backward_lstm_master:evaluate()
  self.attention_master:evaluate()
  
  for i = 1, self.opts.seqLen do
    self.forward_lstms[i]:evaluate()
    self.backward_lstms[i]:evaluate()
    self.attentions[i]:evaluate()
  end
end

-- x, x_mask, x_pos, y
function SelectNet:trainBatch(x, x_mask, infeats, y, sgdParam)
  self:training()
  
  x = self:transData(x)
  x_mask = self:transData(x_mask)
  y = self:transData(y)

  local x_feats = {}
  for i, feat in ipairs(infeats) do
    x_feats[i] = self:transData(feat)
  end
  
  local x_mask_t = x_mask:t()
  local x_mask_sub = (-x_mask_t + 1) * -50
  x_mask_sub = self:transData(x_mask_sub)
  
  local function feval(params_)
    if self.params ~= params_ then
      self.params:copy(params_)
    end
    self.grads:zero()
    
    -- forward pass for forward lstm
    local Tx = x:size(1)
    for i = 1, self.opts.nlayers * 2 do
      self.fwd_hs[0][i]:zero()
    end
    
    self.all_fwd_bak_hs:resize(self.opts.batchSize, Tx, self.opts.nhid * 2)
    for t = 1, Tx do
      local x_in = {}
      table.insert(x_in, x[{t, {}}])
      for i, feat in ipairs(x_feats) do 
        table.insert(x_in, feat[{ t, {} }]) 
      end
      table.insert(x_in, self.fwd_hs[t-1])

      self.fwd_hs[t] = self.forward_lstms[t]:forward(x_in)
      self.all_fwd_bak_hs[{ {}, t, {1, self.opts.nhid} }] = self.fwd_hs[t][self.opts.nlayers * 2]
      if self.opts.useGPU then cutorch.synchronize() end
    end
    
    -- forward pass for backward lstm
    for i = 1, 2 * self.opts.nlayers do
      self.bak_hs[Tx+1][i]:zero()
    end
    for t = Tx, 1, -1 do
      local x_in = {}
      table.insert(x_in, x[{t, {}}])
      for i, feat in ipairs(x_feats) do 
        table.insert(x_in, feat[{ t, {} }]) 
      end
      table.insert(x_in, self.bak_hs[t+1])
      
      self.bak_hs[t] = self.backward_lstms[t]:forward(x_in)
      local cmask = x_mask[{ t, {} }]:view(self.opts.batchSize, 1):expand(self.opts.batchSize, self.opts.nhid)
      for i = 1, 2*self.opts.nlayers do
        self.bak_hs[t][i]:cmul( cmask )
      end
      self.all_fwd_bak_hs[{ {}, t, {self.opts.nhid + 1, 2 * self.opts.nhid} }] = self.bak_hs[t][2 * self.opts.nlayers]
      if self.opts.useGPU then cutorch.synchronize() end
    end
    
    -- forward pass for attention model
    local loss = 0
    local y_preds = {}
    local Ty = y:size(1)
    assert(Ty + 1 == Tx, 'Tx words sentence must have Tx - 1 options')
    for t = 2, Tx do
      y_preds[t] = self.attentions[t]:forward({self.all_fwd_bak_hs, self.all_fwd_bak_hs[{ {}, t, {} }], Tx, 
          x_mask_t, x_mask_sub})
      local loss_ = self.criterions[t]:forward({y_preds[t], y[{ t-1, {} }], self.opts.batchSize})
      loss = loss + loss_
    end
    
    self.df_all_fwd_bak_hs:resize(self.opts.batchSize, Tx, self.opts.nhid * 2):zero()
    -- backward pass for attention model
    for t = Tx, 2, -1 do
      local df_crit = self.criterions[t]:backward({y_preds[t], y[{ t-1, {} }], self.opts.batchSize})
      local tmp_df_all_hs, tmp_df_a_h, _, _, _ = unpack(
        self.attentions[t]:backward({self.all_fwd_bak_hs, self.all_fwd_bak_hs[{ {}, t, {} }], Tx, 
            x_mask_t, x_mask_sub}, df_crit)
      )
      self.df_all_fwd_bak_hs:add( tmp_df_all_hs )
      self.df_all_fwd_bak_hs[{ {}, t, {} }]:add( tmp_df_a_h )
    end
    
    -- prepare backward prop for forward and backward lstms
    for i = 1, 2 * self.opts.nlayers do
      self.df_bak_h[i]:zero()
      self.df_fwd_h[i]:zero()
    end
    
    -- backward pass for backward lstm
    for t = 1, Tx do
      -- no mask is needed, since in the forward pass, some rows of self.bak_hs[t+1] have been set to 0
      -- No error will be back-prop
      self.df_bak_h[2 * self.opts.nlayers]:add( self.df_all_fwd_bak_hs[{ {}, t, {self.opts.nhid + 1, 2 * self.opts.nhid} }] )
      local x_in = {}
      table.insert(x_in, x[{t, {}}])
      for i, feat in ipairs(x_feats) do 
        table.insert(x_in, feat[{ t, {} }]) 
      end
      table.insert(x_in, self.bak_hs[t+1])
      for _, feat in ipairs(x_in) do
      end
      local tmp = self.backward_lstms[t]:backward(x_in, self.df_bak_h)
      model_utils.copy_table(self.df_bak_h, tmp[#tmp])
    end
    
    -- backward pass for forward lstm
    for t = Tx, 1, -1 do
      self.df_fwd_h[2*self.opts.nlayers]:add( self.df_all_fwd_bak_hs[{ {}, t, {1, self.opts.nhid} }] )
      -- mask should be used here
      local cmask = x_mask[{ t, {} }]:view(self.opts.batchSize, 1):expand(self.opts.batchSize, self.opts.nhid)
      for i = 1, 2*self.opts.nlayers do
        self.df_fwd_h[i]:cmul( cmask )
      end
      local x_in = {}
      table.insert(x_in, x[{t, {}}])
      for i, feat in ipairs(x_feats) do 
        table.insert(x_in, feat[{ t, {} }]) 
      end
      table.insert(x_in, self.fwd_hs[t-1])
      local tmp = self.forward_lstms[t]:backward(x_in, self.df_fwd_h )
      model_utils.copy_table(self.df_fwd_h, tmp[#tmp])
    end
    
    if self.opts.embedOption ~= nil and self.opts.embedOption == 'fineTune' then
      self.net_lookup:applyGradMask()
    end
    
    if self.opts.gradClip < 0 then
      local clip = -self.opts.gradClip
      self.grads:clamp(-clip, clip)
    elseif self.opts.gradClip > 0 then
      local maxGradNorm = self.opts.gradClip
      local gradNorm = self.grads:norm()
      if gradNorm > maxGradNorm then
        local shrinkFactor = maxGradNorm / gradNorm
        self.grads:mul(shrinkFactor)
      end
    end
    
    return loss, self.grads
  end
  
  local _, loss_ = self.optimMethod(feval, self.params, sgdParam)
  
  return loss_[1]
end

-- x, x_mask, x_pos, y
function SelectNet:validBatch(x, x_mask, infeats, y)
  self:evaluate()
  x = self:transData(x)
  x_mask = self:transData(x_mask)
  local x_feats = {}
  for i, feat in ipairs(infeats) do
    x_feats[i] = self:transData(feat)
  end
  y = self:transData(y)

  local x_mask_t = x_mask:t()
  local x_mask_sub = (-x_mask_t + 1) * -50
  x_mask_sub = self:transData( x_mask_sub )
  
  -- forward pass for forward lstm
  local Tx = x:size(1)
  for i = 1, self.opts.nlayers * 2 do
    self.fwd_hs[0][i]:zero()
  end
  self.all_fwd_bak_hs:resize(self.opts.batchSize, Tx, self.opts.nhid * 2)
  for t = 1, Tx do
    local x_in = {}
    table.insert(x_in, x[{t, {}}])
    for i, feat in ipairs(x_feats) do 
      table.insert(x_in, feat[{ t, {} }]) 
    end
    table.insert(x_in, self.fwd_hs[t-1])
    self.fwd_hs[t] = self.forward_lstms[t]:forward(x_in)
    self.all_fwd_bak_hs[{ {}, t, {1, self.opts.nhid} }] = self.fwd_hs[t][self.opts.nlayers*2]
    if self.opts.useGPU then cutorch.synchronize() end
  end
  
  -- forward pass for backward lstm
  for i = 1, 2 * self.opts.nlayers do
    self.bak_hs[Tx+1][i]:zero()
  end
  for t = Tx, 1, -1 do
    local x_in = {}
    table.insert(x_in, x[{t, {}}])
    for i, feat in ipairs(x_feats) do 
      table.insert(x_in, feat[{ t, {} }]) 
    end
    table.insert(x_in, self.bak_hs[t+1])
    self.bak_hs[t] = self.backward_lstms[t]:forward(x_in)
    local cmask = x_mask[{ t, {} }]:view(self.opts.batchSize, 1):expand(self.opts.batchSize, self.opts.nhid)
    for i = 1, 2 * self.opts.nlayers do
      self.bak_hs[t][i]:cmul( cmask )
    end
    self.all_fwd_bak_hs[{ {}, t, {self.opts.nhid + 1, 2 * self.opts.nhid} }] = self.bak_hs[t][2 * self.opts.nlayers]
    if self.opts.useGPU then cutorch.synchronize() end
  end
  
  -- forward pass for attention model
  local loss = 0
  local y_preds = {}
  local Ty = y:size(1)
  assert(Ty + 1 == Tx, 'Tx words sentence must have Tx - 1 options')
  for t = 2, Tx do
    y_preds[t] = self.attentions[t]:forward({self.all_fwd_bak_hs, self.all_fwd_bak_hs[{ {}, t, {} }], Tx, 
        x_mask_t, x_mask_sub})
    local loss_ = self.criterions[t]:forward({y_preds[t], y[{ t-1, {} }], self.opts.batchSize})
    loss = loss + loss_
  end
  
  return loss, y_preds
end
