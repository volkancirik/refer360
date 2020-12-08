'''Unimodal baselines.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet_3 import Unet3
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from model_utils import vectorize_seq, PositionalEncoding, FOV_EMB_SIZE, build_fov_embedding


class VisionOnly(nn.Module):
  """
  Vision only unimodal baseline
  """

  def __init__(self, batch_size,
               hidden_size=128,
               n_actions=5,
               w=400,
               h=400,
               ch=1,
               n_objects=0,
               use_queries=False,
               fov_emb_mode=2):
    super(VisionOnly, self).__init__()

    self.unet = Unet3(2048, 4, hidden_size).cuda().eval()

    # actor's layer
    self.n_actions = n_actions
    self.w = w
    self.h = h
    img_size = w*h*ch
    self.img_size = img_size
    self.bn = nn.BatchNorm2d(4)
    self.action_head0 = nn.Linear(img_size, hidden_size * 4)
    self.action_head1 = nn.Linear(hidden_size*4, hidden_size*2)
    self.action_head2 = nn.Linear(hidden_size*2, hidden_size)
    self.action_head3 = nn.Linear(hidden_size, n_actions)

    # critic's layer
    self.value_head = nn.Linear(img_size, 1)

    # action & reward buffer
    self.rewards = [[] for bb in range(batch_size)]
    self.saved_actions = [[] for bb in range(batch_size)]
    self.batch_size = batch_size
    self.use_queries = use_queries
    self.n_objects = n_objects
    if self.use_queries:
      self.obj_emb = nn.Embedding(self.n_objects, hidden_size * 4)

    self.fov_emb_mode = fov_emb_mode

    if self.fov_emb_mode > 0:
      self.fov2logit = nn.Linear(FOV_EMB_SIZE, hidden_size)
      self.fov2logit.bias.data.fill_(0)
    self.dropout = nn.Dropout(p=0.1)

  def forward(self, state):
    """
    forward of both actor and critic
    """
    x = state['im_batch']
    pred = self.unet(x).squeeze(1)
    #bn_pred = self.bn(pred)

    resized_pred = pred.reshape(pred.size(0), self.img_size)
    logits1 = self.dropout(F.relu(self.action_head0(resized_pred)))
    if self.use_queries:
      emb = self.dropout(self.obj_emb(state['queries']).squeeze(1))
      logits1 = logits1 + emb

    logits2 = self.dropout(F.relu(self.action_head1(logits1)))
    logits3 = self.dropout(F.relu(self.action_head2(logits2)))

    if self.fov_emb_mode == 0:
      pass
    elif self.fov_emb_mode == 1:
      fov_emb = build_fov_embedding(state['latitude'], state['longitude'])
      logits3 = self.fov2logit(fov_emb)
    elif self.fov_emb_mode == 2:
      fov_emb = build_fov_embedding(state['latitude'], state['longitude'])
      logits3 = self.fov2logit(fov_emb) + logits3
    else:
      raise NotImplementedError()

    logits = self.action_head3(logits3)

    # if self.use_queries:
    #   logits = logits4
    # else:
    #   logits = torch.sigmoid(logits4)

    state_values = self.value_head(resized_pred)
    loc_pred = F.softmax(
        resized_pred, dim=-1).reshape(pred.size(0), self.w, self.h)

    out = {}
    out['action_logits'] = logits
    out['state_values'] = state_values
    out['loc_pred'] = loc_pred
    return out


class TextOnly(nn.Module):
  """
  Text only unimodal baseline
  """

  def __init__(self, vocab, n_emb, n_head, n_hid, n_layers, batch_size, dropout=0.5, n_actions=5):
    super(TextOnly, self).__init__()

    self.vocab = vocab
    self.n_actions = n_actions
    self.n_vocab = len(vocab)
    self.n_emb = n_emb
    self.n_head = n_head
    self.n_hid = n_hid
    self.n_layers = n_layers

    # action & reward buffer
    self.rewards = [[] for bb in range(batch_size)]
    self.saved_actions = [[] for bb in range(batch_size)]
    self.batch_size = batch_size

    self.pos_encoder = PositionalEncoding(n_emb, dropout)
    encoder_layers = TransformerEncoderLayer(n_emb, n_head, n_hid, dropout)
    self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
    self.encoder = nn.Embedding(self.n_vocab, n_emb)

    # enc -> location prediction
    self.enc2pred = nn.Linear(n_emb, 400 * 400)
    # actor's layer
    self.action_head0 = nn.Linear(n_emb, 512)
    self.action_head1 = nn.Linear(512, 256)
    self.action_head2 = nn.Linear(256, 128)
    self.action_head3 = nn.Linear(128, self.n_actions)

    # critic's layer
    self.value_head = nn.Linear(n_emb, 1)

  def forward(self, state):
    """
    forward of both actor and critic
    """
    instruction = [' '.join(list(sum(o['refexps'], [])))
                   for o in state['observations']]

    x, x_mask, x_emb_mask, _ = vectorize_seq(
        instruction, self.vocab, self.n_emb, cuda=True, permute=True)

    src = self.encoder(x)
    src = self.pos_encoder(src)

    encoding = self.transformer_encoder(
        src, src_key_padding_mask=x_mask)
    encoding = encoding * x_emb_mask
    encoding = torch.sum(encoding, dim=0)

    pred = self.enc2pred(encoding)

    logits1 = F.relu(self.action_head0(encoding))
    logits2 = F.relu(self.action_head1(logits1))
    logits3 = F.relu(self.action_head2(logits2))
    logits = self.action_head3(logits3)

    state_values = self.value_head(encoding)
    loc_pred = F.softmax(pred, dim=-1).reshape(pred.size(0), 400, 400)

    out = {}
    out['action_logits'] = logits
    out['state_values'] = state_values
    out['loc_pred'] = loc_pred
    return out
