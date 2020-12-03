'''Finder is a model with actor, memory, and localizer modules.
'''
import torchvision.models as models
import torch
import torch.nn as nn
from models.td_models import clones
from model_utils import vectorize_seq
LOG_ZERO_PROB = -10000

from get_localizer import get_localizer


class Actor(nn.Module):
  '''The module for parameterizing the action prediction.
  '''

  def __init__(self, n_inp, n_hid, n_layers, n_actions=5):
    super(Actor, self).__init__()

    self.inp2act = nn.Linear(n_inp, n_hid)
    self.inp2val = nn.Linear(n_inp, n_hid)

    self.act = clones(nn.Linear(n_hid, n_hid), n_layers)
    self.val = clones(nn.Linear(n_hid, n_hid), n_layers)

    self.act2out = nn.Linear(n_hid, n_actions)
    self.val2out = nn.Linear(n_hid, 1)
    self.act2out.bias.data.fill_(0)
    self.val2out.bias.data.fill_(0)

    self.RELU = nn.ReLU()
    self.n_layers = n_layers

  def forward(self, inp):
    act_inp = self.RELU(self.inp2act(inp))
    val_inp = self.RELU(self.inp2val(inp))

    for ii in range(self.n_layers):
      act_inp = self.RELU(self.act[ii](act_inp))
      val_inp = self.RELU(self.val[ii](val_inp))

    return self.act2out(act_inp), self.val2out(val_inp)


class FinderLookAhead(nn.Module):
  """Finder with localizer, actor, and memory modules.
  """

  def __init__(self, args, vocab, n_actions=5):

    super(FinderLookAhead, self).__init__()
    self.vocab = vocab
    self.n_actions = n_actions
    self.n_vocab = len(vocab)
    self.n_emb = args.n_emb
    self.n_hid = args.n_hid
    self.n_layers = args.n_layers

    self.cnn = nn.Sequential(
        *(list(models.resnet18(pretrained=True).children())[:-args.cnn_layer])).cuda().eval()

    self.localizer = get_localizer(args, self.n_vocab)

    self.obs2state = nn.Linear(args.n_obs, args.n_hid)
    self.memory = nn.GRU(args.n_hid, args.n_hid, args.n_layers)
    for name, param in self.memory.named_parameters():
      if 'bias' in name:
        nn.init.constant_(param, 0)
      elif 'weight' in name:
        nn.init.orthogonal_(param)
    self.actor = Actor(args.n_hid, args.n_hid, args.n_layers,
                       n_actions=n_actions)

    self.rewards = [[] for bb in range(args.batch_size)]
    self.saved_actions = [[] for bb in range(args.batch_size)]
    self.batch_size = args.batch_size
    self.use_masks = args.use_masks
    self.use_raw_image = args.use_raw_image

    self.neighbor2pred = nn.Linear(64, 1)
    self.act2out = nn.Linear(10000, 1)
    self.loc2pred = nn.Linear(10000, 1)
    self.val2out = nn.Linear(n_actions, 1)

  def forward(self, observations, rnn_hidden=None):

    instruction = [' '.join(list(sum(o['refexps'], [])))
                   for o in observations['observations']]

    texts, texts_mask, texts_emb_mask, seq_lengths = vectorize_seq(
        instruction, self.vocab, self.n_emb, cuda=True, permute=self.use_masks)

    # return text_embed with localizer?
    kwargs = {}
    if self.use_masks:
      kwargs['texts_mask'] = texts_mask
      kwargs['texts_emb_mask'] = texts_emb_mask

    if self.use_raw_image:
      kwargs['im_batch'] = observations['im_batch']
      kwargs['imgs'] = [o['img']
                        for o in observations['observations']]
      kwargs['instruction'] = instruction
      loc_pred = self.localizer(kwargs)
    else:
      images = self.cnn(observations['im_batch'])
      loc_pred = self.localizer(images, texts, seq_lengths.cpu(), **kwargs)

    obs = loc_pred.reshape(
        self.batch_size, loc_pred.size(1) * loc_pred.size(2))

    neighbors = [o['neighbors']
                 for o in observations['observations']]
    all_neighbors = torch.cat(neighbors, 0)
    neighbors_feat = self.cnn(all_neighbors)
    neighbors_flat = neighbors_feat.reshape(
        self.batch_size * 4, 64, loc_pred.size(1) * loc_pred.size(2)).permute(0, 2, 1)

    neighbors_pred = self.neighbor2pred(neighbors_flat).squeeze()
    neighbors_stack = neighbors_pred.reshape(self.batch_size, 4, 10000)
    obs_stack = obs.unsqueeze(1).repeat(1, 4, 1)

    fused = neighbors_stack * obs_stack
    neighbors_scores = self.act2out(fused).squeeze(2)
    pred_score = self.loc2pred(obs)

    action_logits = torch.cat([pred_score, neighbors_scores], 1)
    state_values = self.val2out(action_logits)

    # state = self.obs2state(obs)
    # state_now, rnn_hidden = self.memory(
    #     state.unsqueeze(0), rnn_hidden)
    # state_now = state_now.squeeze(0)

    # action_logits,a state_values = self.actor(state_now, neighbors)

    out = {}
    out['action_logits'] = action_logits
    out['state_values'] = state_values
    out['loc_pred'] = loc_pred
    out['rnn_hidden'] = None

    return out
