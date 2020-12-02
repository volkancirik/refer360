import torch
import torchvision.models as models
import torch.nn as nn
import os

from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

import numpy as np

# Max length including <bos> and <eos>
MAX_SEQ_LENGTH = 100
DATA_PATH = '../py_bottom_up_attention/demo/data/genome/1600-400-20'
DETECTRON2_YAML = '../py_bottom_up_attention/configs/VG-Detection/faster_rcnn_R_101_C4_caffe.yaml'

from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs
from torchvision.ops import nms
from detectron2.structures import Boxes, Instances

MIN_BOXES = 36
MAX_BOXES = 36

BOXES = [[x, y, x+16, y + 16]
         for x in range(0, 400, 16) for y in range(0, 400, 16)]


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
  idxs = torch.arange(num_objs).cuda() * num_bbox_reg_classes + max_classes
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


class LXMERTLocalizer(nn.Module):
  def __init__(self, args, num_logits=100*100):
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

    self.use_detectron = args.use_detectron

    if self.use_detectron:
      print('Detectron will be used.')
      data_path = DATA_PATH

      vg_classes = []
      with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
        for object in f.readlines():
          vg_classes.append(object.split(',')[0].lower().strip())

      MetadataCatalog.get("vg").thing_classes = vg_classes
      yaml_file = DETECTRON2_YAML
      cfg = get_cfg()
      cfg.merge_from_file(yaml_file
                          )
      cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
      cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
      cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
      # VG Weight
      cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe.pkl"
      self.predictor = DefaultPredictor(cfg)
    else:
      print('Resnet will be used.')
      self.cnn = nn.Sequential(
          *(list(models.resnet18(pretrained=True).children())[:-3])).cuda().eval()
      self.cnn2box = nn.Linear(256, 2048)

  def get_det2_boxes(self, imgs):

    instances_list, features_list = extract_detectron2_features(
        self.predictor, imgs)

    feats = []
    boxes = []
    for instances, features in zip(instances_list, features_list):
      feats.append(features.unsqueeze(0))
      boxes.append(instances.pred_boxes.tensor.unsqueeze(0))

    feats = torch.cat(feats, 0)
    boxes = torch.cat(boxes, 0)

    boxes[..., (0, 2)] /= 400.
    boxes[..., (1, 3)] /= 400.
    return feats, boxes

  def get_cnn_boxes(self, im_batch):
    batch_size = im_batch.size(0)

    regions = self.cnn(im_batch)

    feats = self.cnn2box(regions.view(
        batch_size, 256, 25*25).permute(0, 2, 1))

    boxes = torch.tensor(BOXES).unsqueeze(
        0).repeat(feats.size(0), 1, 1).float().cuda()
    boxes[..., (0, 2)] /= 400.
    boxes[..., (1, 3)] /= 400.

    return feats, boxes

  def forward(self, inp):

    instruction = inp['instruction']

    if self.use_detectron:
      feats, boxes = self.get_det2_boxes(inp['imgs'])
    else:
      feats, boxes = self.get_cnn_boxes(inp['im_batch'])

    x = self.lxrt_encoder(instruction, (feats, boxes))
    batch_size = x.size(0)
    logit = self.logit_fc(x).view(batch_size, 100, 100)

    return logit
