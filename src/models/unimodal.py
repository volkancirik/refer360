'''Unimodal baselines.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unet_3 import Unet3
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from src.utils import PositionalEncoding, vectorize_seq


class VisionOnly(nn.Module):
  """
  Vision only unimodal baseline
  """

  def __init__(self, batch_size, n_actions=5):
    super(VisionOnly, self).__init__()

    self.unet = Unet3(3, 1, 128).cuda().eval()

    # actor's layer
    self.n_actions = n_actions
    self.action_head0 = nn.Linear(400*400, 512)
    self.action_head1 = nn.Linear(512, 256)
    self.action_head2 = nn.Linear(256, 128)
    self.action_head3 = nn.Linear(128, n_actions)

    # critic's layer
    self.value_head = nn.Linear(400*400, 1)

    # action & reward buffer
    self.rewards = [[] for bb in range(batch_size)]
    self.saved_actions = [[] for bb in range(batch_size)]
    self.batch_size = batch_size

  def forward(self, state):
    """
    forward of both actor and critic
    """
    x = state['im_batch']
    pred = self.unet(x).squeeze(1)

    # resized_pred = self.resnet(x).squeeze(2).squeeze(2)
    resized_pred = pred.reshape(pred.size(0), 400*400)

    logits1 = F.relu(self.action_head0(resized_pred))
    logits2 = F.relu(self.action_head1(logits1))
    logits3 = F.relu(self.action_head2(logits2))
    logits = self.action_head3(logits3)

    state_values = self.value_head(resized_pred)
    loc_pred = F.softmax(resized_pred, dim=-1).reshape(pred.size(0), 400, 400)

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
