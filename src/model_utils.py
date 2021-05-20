'''Utils for models and manipulating tensors.
'''
import torch
import torch.nn as nn
#import torch.nn.functional as F
import math
import base64
import time
import csv
import sys


from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs
from torchvision.ops import nms
from detectron2.structures import Boxes, Instances
import numpy as np

import io
import PIL.Image
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MIN_BOXES = 36
MAX_BOXES = 36

FOV_EMB_SIZE = 128

# BUTD features fields for FoVs
csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]


def coord_gaussian(x, y, gt_x, gt_y, sigma):
  '''Return the gaussian-weight of a pixel based on distance.
  '''
  pixel_val = (1 / (2 * np.pi * sigma ** 2)) * \
      np.exp(-((x - gt_x) ** 2 + (y - gt_y) ** 2) / (2 * sigma ** 2))
  return pixel_val if pixel_val > 1e-5 else 0.0


def gaussian_target(gt_x, gt_y, sigma=3.0,
                    width=400, height=400):
  '''Based on https://github.com/lil-lab/touchdown/tree/master/sdr
  '''
  target = torch.tensor([[coord_gaussian(x, y, gt_x, gt_y, sigma)
                          for x in range(width)] for y in range(height)]).float().to(DEVICE)
  return target


def smoothed_gaussian(pred, gt_x, gt_y,
                      sigma=3.0,
                      height=400, width=400):
  '''KLDivLoss for pixel prediction.
  '''

  loss_func = nn.KLDivLoss(reduction='sum')

  target = gaussian_target(gt_x, gt_y,
                           sigma=sigma, width=width, height=height)
  loss = loss_func(pred.unsqueeze(0), target.unsqueeze(0))
  return loss


def get_det2_features(detections):
  boxes = []
  obj_classes = []
  scores = []
  for box, obj_class, score in zip(detections.pred_boxes, detections.pred_classes, detections.scores):
    box = box.cpu().detach().numpy()
    boxes.append(box)
    obj_class = obj_class.cpu().detach().numpy().tolist()
    obj_classes.append(obj_class)
    scores.append(score.item())

  return boxes, obj_classes, scores


class PositionalEncoding(nn.Module):
  "Implement the PE function."

  def __init__(self, d_model, dropout, max_len=5000):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)

    # Compute the positional encodings once in log space.
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                         -(math.log(10000.0) / d_model))

    size_sin = pe[:, 0::2].size()
    pe[:, 0::2] = torch.sin(position * div_term)[:size_sin[0], : size_sin[1]]
    size_cos = pe[:, 1::2].size()
    pe[:, 1::2] = torch.cos(position * div_term)[:size_cos[0], : size_cos[1]]
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)

  def forward(self, x):
    pos_info = self.pe[:, : x.size(1)].clone().detach().requires_grad_(True)
    x = x + pos_info
    return self.dropout(x)


def load_obj_tsv(fname, topk=None):
  """Source: https://github.com/airsplay/lxmert/blob/f65a390a9fd3130ef038b3cd42fe740190d1c9d2/src/utils.py#L16
  Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
  data = []
  start_time = time.time()
  print("Start to load Faster-RCNN detected objects from %s" % fname)
  with open(fname) as f:
    reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
    for i, item in enumerate(reader):

      for key in ['img_h', 'img_w', 'num_boxes']:
        item[key] = int(item[key])

      boxes = item['num_boxes']
      decode_config = [
          ('objects_id', (boxes, ), np.int64),
          ('objects_conf', (boxes, ), np.float32),
          ('attrs_id', (boxes, ), np.int64),
          ('attrs_conf', (boxes, ), np.float32),
          ('boxes', (boxes, 4), np.float32),
          ('features', (boxes, -1), np.float32),
      ]
      for key, shape, dtype in decode_config:
        item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
        item[key] = item[key].reshape(shape)
        item[key].setflags(write=False)

      data.append(item)
      if topk is not None and len(data) == topk:
        break
  elapsed_time = time.time() - start_time
  print("Loaded %d images in file %s in %d seconds." %
        (len(data), fname, elapsed_time))
  return data


def build_fov_embedding(latitude, longitude):
  """
  Position embedding:
  latitude 64D + longitude 64D
  1) latitude: [sin(latitude) for _ in range(1, 33)] +
  [cos(latitude) for _ in range(1, 33)]
  2) longitude: [sin(longitude) for _ in range(1, 33)] +
  [cos(longitude) for _ in range(1, 33)]
  """
  quarter = int(FOV_EMB_SIZE / 4)

  embedding = torch.zeros(latitude.size(0), FOV_EMB_SIZE).to(DEVICE)
  embedding[:,  0:quarter*1] = torch.sin(latitude)
  embedding[:, quarter*1:quarter*2] = torch.cos(latitude)
  embedding[:, quarter*2:quarter*3] = torch.sin(longitude)
  embedding[:, quarter*3:quarter*4] = torch.cos(longitude)

  return embedding


def weights_init_uniform_rule(m):
  classname = m.__class__.__name__
  # for every Linear layer in a model..
  if classname.find('Linear') != -1:
    # get the number of the inputs
    n = m.in_features
    y = 1.0/np.sqrt(n)
    m.weight.data.uniform_(-y, y)
    m.bias.data.fill_(0)


def fast_rcnn_inference_single_image(
    boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image
):
  scores = scores[:, :-1]
  num_bbox_reg_classes = boxes.shape[1] // 4
  # Convert to Boxes to use the `clip` function ...
  boxes = Boxes(boxes.reshape(-1, 4))
  boxes.clip(image_shape)
  boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

  # Select max scores
  max_scores, max_classes = scores.max(1)       # R x C --> R
  num_objs = boxes.size(0)
  boxes = boxes.view(-1, 4)
  idxs = torch.arange(num_objs).to(DEVICE) * num_bbox_reg_classes + max_classes
  max_boxes = boxes[idxs]     # Select max boxes according to the max scores.

  # Apply NMS
  keep = nms(max_boxes, max_scores, nms_thresh)
  if topk_per_image >= 0:
    keep = keep[:topk_per_image]
  boxes, scores = max_boxes[keep], max_scores[keep]

  result = Instances(image_shape)
  result.pred_boxes = Boxes(boxes)
  result.scores = scores
  result.pred_classes = max_classes[keep]

  return result, keep


def extract_detectron2_features(detector, raw_images):
  with torch.no_grad():
    # Preprocessing
    inputs = []
    for raw_image in raw_images:
      image = detector.transform_gen.get_transform(
          raw_image).apply_image(raw_image)
      image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
      inputs.append(
          {"image": image, "height": raw_image.shape[0], "width": raw_image.shape[1]})
    images = detector.model.preprocess_image(inputs)

    # Run Backbone Res1-Res4
    features = detector.model.backbone(images.tensor)

    # Generate proposals with RPN
    proposals, _ = detector.model.proposal_generator(images, features, None)

    # Run RoI head for each proposal (RoI Pooling + Res5)
    proposal_boxes = [x.proposal_boxes for x in proposals]
    features = [features[f] for f in detector.model.roi_heads.in_features]
    box_features = detector.model.roi_heads._shared_roi_transform(
        features, proposal_boxes
    )
    # (sum_proposals, 2048), pooled to 1x1
    feature_pooled = box_features.mean(dim=[2, 3])

    # Predict classes and boxes for each proposal.
    pred_class_logits, pred_proposal_deltas = detector.model.roi_heads.box_predictor(
        feature_pooled)
    rcnn_outputs = FastRCNNOutputs(
        detector.model.roi_heads.box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        detector.model.roi_heads.smooth_l1_beta,
    )

    # Fixed-number NMS
    instances_list, ids_list = [], []
    probs_list = rcnn_outputs.predict_probs()
    boxes_list = rcnn_outputs.predict_boxes()
    for probs, boxes, image_size in zip(probs_list, boxes_list, images.image_sizes):
      for nms_thresh in np.arange(0.3, 1.0, 0.1):
        instances, ids = fast_rcnn_inference_single_image(
            boxes, probs, image_size,
            score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=MAX_BOXES
        )
        if len(ids) >= MIN_BOXES:
          break
      instances_list.append(instances)
      ids_list.append(ids)

    # Post processing for features
    # (sum_proposals, 2048) --> [(p1, 2048), (p2, 2048), ..., (pn, 2048)]
    features_list = feature_pooled.split(rcnn_outputs.num_preds_per_image)
    roi_features_list = []
    for ids, features in zip(ids_list, features_list):
      roi_features_list.append(features[ids].detach())

    # Post processing for bounding boxes (rescale to raw_image)
    raw_instances_list = []
    for instances, input_per_image, image_size in zip(
        instances_list, inputs, images.image_sizes
    ):
      height = input_per_image.get("height", image_size[0])
      width = input_per_image.get("width", image_size[1])
      raw_instances = detector_postprocess(instances, height, width)
      raw_instances_list.append(raw_instances)

    return raw_instances_list, roi_features_list


def compute_precision_with_logits(logits, labels_vector,
                                  precision_k=1,
                                  mask=None):
  labels = torch.zeros(*logits.size()).to(DEVICE)
  for ll in range(logits.size(0)):
    labels[ll, :].scatter_(0, labels_vector[ll], 1)

  adjust_score = False
  if type(mask) != type(None):
    labels = labels * mask.unsqueeze(0).expand(labels.size())
    adjust_score = True
  one_hots = torch.zeros(*labels.size()).to(DEVICE)
  logits = torch.sort(logits, 1, descending=True)[1]
  one_hots.scatter_(1, logits[:, :precision_k], 1)

  batch_size = logits.size(0)
  scores = ((one_hots * labels).sum(1) >= 1).float().sum() / batch_size
  if adjust_score:
    valid_rows = (labels.sum(1) > 0).sum(0)
    if valid_rows == 0:
      scores = torch.tensor(1.0).to(DEVICE)
    else:
      hit = (scores * batch_size)
      scores = hit / valid_rows
  return scores


def vectorize_seq(sequences, vocab, emb_dim, cuda=False, permute=False):
  '''Generates one-hot vectors and masks for list of sequences.
  '''
  vectorized_seqs = [vocab.encode(sequence) for sequence in sequences]

  seq_lengths = torch.tensor(list(map(len, vectorized_seqs)))
  seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
  max_length = seq_tensor.size(1)

  emb_mask = torch.zeros(len(seq_lengths), max_length, emb_dim)
  for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
    seq_tensor[idx, :seqlen] = torch.tensor(seq)
    emb_mask[idx, :seqlen, :] = 1.

  idx = torch.arange(max_length).unsqueeze(0).expand(seq_tensor.size())
  len_expanded = seq_lengths.unsqueeze(1).expand(seq_tensor.size())
  mask = (idx >= len_expanded)

  if permute:
    seq_tensor = seq_tensor.permute(1, 0)
    emb_mask = emb_mask.permute(1, 0, 2)
  if cuda:
    return seq_tensor.to(DEVICE), mask.to(DEVICE), emb_mask.to(DEVICE), seq_lengths.to(DEVICE)
  return seq_tensor, mask, emb_mask, seq_lengths


class F1_Loss(nn.Module):
  '''Calculate F1 score. Can work with gpu tensors

  The original implmentation is written by Michal Haltuf on Kaggle.

  Returns
  -------
  torch.Tensor
      `ndim` == 1. epsilon <= val <= 1

  Reference
  ---------
  - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
  # sklearn.metrics.f1_score
  - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
  - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
  - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
  '''

  def __init__(self, epsilon=1e-7,
               num_classes=2):
    super().__init__()
    self.epsilon = epsilon
    self.num_classes = num_classes

  def forward(self, y_pred, y_true,):
    assert y_pred.ndim == 2
    assert y_true.ndim == 1
    y_true = F.one_hot(y_true, self.num_classes).to(torch.float32)
    y_pred = F.softmax(y_pred, dim=1)

    tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
    # tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

    precision = tp / (tp + fp + self.epsilon)
    recall = tp / (tp + fn + self.epsilon)

    f1 = 2 * (precision*recall) / (precision + recall + self.epsilon)
    f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
    return 1 - f1.mean()


class F1_Binary_Loss(nn.Module):
  '''F1 for binary classification.
  '''

  def __init__(self, epsilon=1e-7):
    super().__init__()
    self.epsilon = epsilon

  def forward(self, y_pred, y_true):
    assert y_pred.ndim == 2
    assert y_true.ndim == 2

    y_true = y_true.to(torch.float32)
    y_pred = y_pred.to(torch.float32)

    tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

    precision = tp / (tp + fp + self.epsilon)
    recall = tp / (tp + fn + self.epsilon)

    f1 = 2 * (precision*recall) / (precision + recall + self.epsilon)
    f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)

    return f1.mean()


def get_confusion_matrix_image(labels, matrix, title='Title', tight=False, cmap=cm.copper):

  fig, ax = plt.subplots()
  _ = ax.imshow(matrix, cmap=cmap)

  ax.set_xticks(np.arange(len(labels)))
  ax.set_yticks(np.arange(len(labels)))
  ax.set_xticklabels(labels)
  ax.set_yticklabels(labels)

  # Rotate the tick labels and set their alignment.
  plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
           rotation_mode="anchor")

  # Loop over data dimensions and create text annotations.

  for i in range(len(labels)):
    for j in range(len(labels)):

      _ = ax.text(j, i, '{:0.2f}'.format(matrix[i, j]),
                  ha="center", va="center", color="w")

  ax.set_title(title)
  if tight:
    fig.tight_layout()
  buf = io.BytesIO()
  plt.savefig(buf, format='jpeg')
  buf.seek(0)
  image = PIL.Image.open(buf)
  image = ToTensor()(image)
  plt.close('all')
  return image
