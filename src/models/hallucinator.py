import torch.nn as nn
import torch.nn.functional as F
from model_utils import FOV_EMB_SIZE
from model_utils import build_fov_embedding
from models.unet_3 import Unet3


class Hallucinator(nn.Module):
  """
  Vision only unimodal baseline
  """

  def __init__(self, batch_size,
               hidden_size=128,
               n_actions=5,
               n_objects=0,
               use_queries=False,
               fov_emb_mode=2):
    super(Hallucinator, self).__init__()

    self.n_actions = n_actions
    self.bn = nn.BatchNorm2d(4)
    self.fmap2score = nn.Linear(2048, 1)
    self.score2hidden = nn.Linear(169, hidden_size*8)

    self.ff_layer0 = nn.Linear(hidden_size*8, hidden_size * 4)
    self.ff_layer1 = nn.Linear(hidden_size*4, hidden_size*2)
    self.ff_layer2 = nn.Linear(hidden_size*2, hidden_size)
    self.ff_layer3 = nn.Linear(hidden_size, n_actions)

    self.batch_size = batch_size
    self.use_queries = use_queries
    self.n_objects = n_objects
    if self.use_queries:
      self.obj_emb = nn.Embedding(self.n_objects, hidden_size * 4)

    self.fov_emb_mode = fov_emb_mode

    if self.fov_emb_mode > 0:
      self.fov2logit = nn.Linear(FOV_EMB_SIZE, hidden_size)
      self.fov2logit.bias.data.fill_(1)
    self.dropout = nn.Dropout(p=0.1)
    self.SIGMOID = nn.Sigmoid()

  def forward(self, state):
    """
    forward of both actor and critic
    """
    x = state['im_batch']
    x = x.reshape(x.size(0), x.size(1), x.size(2)*x.size(3))
    x = x.permute(0, 2, 1)

    fmap_score = self.dropout(F.relu(self.fmap2score(x))).squeeze(2)
    #bn_pred = self.bn(pred)
    hid = self.dropout(F.relu(self.score2hidden(fmap_score)))

    logits1 = self.dropout(F.relu(self.ff_layer0(hid)))
    if self.use_queries:
      emb = self.dropout(self.obj_emb(state['queries']).squeeze(1))
      logits1 = logits1 + emb

    logits2 = self.dropout(F.relu(self.ff_layer1(logits1)))
    logits3 = self.dropout(F.relu(self.ff_layer2(logits2)))

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

    logits = self.SIGMOID(self.ff_layer3(logits3))

    out = {}
    out['action_logits'] = logits

    return out


from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU

# Max length including <bos> and <eos>
MAX_SEQ_LENGTH = 40


class LXMERTHallucinator(nn.Module):
  def __init__(self, args, num_logits):
    super().__init__()

    # Build LXRT encoder
    self.lxrt_encoder = LXRTEncoder(
        args,
        max_seq_length=MAX_SEQ_LENGTH
    )
    hid_dim = self.lxrt_encoder.dim
    self.n_actions = num_logits

    self.logit_fc = nn.Sequential(
        nn.Linear(hid_dim, hid_dim * 2),
        GeLU(),
        BertLayerNorm(hid_dim * 2, eps=1e-12),
        nn.Linear(hid_dim * 2, num_logits)
    )
    self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

  def forward(self, inp):
    query = inp['query_strings']

    feats = inp['obj_feats']
    boxes = inp['obj_boxes']
    x = self.lxrt_encoder(query, (feats, boxes))

    logit = self.logit_fc(x)

    out = {}
    out['action_logits'] = logit

    return out
