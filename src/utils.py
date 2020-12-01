"""Utils for io, tokenization etc.
"""
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import math

import os
from tqdm import tqdm

from panoramic_camera_gpu import PanoramicCameraGPU as camera
#from panoramic_camera import PanoramicCamera as camera


import cv2

import io
import PIL.Image
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor

import csv
import base64
import time
import sys
csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]


CAT2LOC = {'restaurant': 'indoor',
           'shop': 'indoor',
           'expo_showroom': 'indoor',
           'living_room': 'indoor',
           'bedroom': 'indoor',
           'street': 'outdoor',
           'plaza_courtyard': 'outdoor',
           }
SPLITS = [
    'validation.seen',
    'validation.unseen',
    'test.seen',
    'test.unseen',
    'train',
]
DIRECTIONS = {
    'canonical': ['up', 'down', 'left', 'right'],
    'cartesian': ['vertical', 'horizontal'],
    'lup': ['lateral', 'up', 'down'],
    'canonical_proximity': ['close_up', 'close_down', 'close_left', 'close_right',
                            'far_up', 'far_down', 'far_left', 'far_right']
}
FOV_EMB_SIZE = 128


def weights_init_uniform_rule(m):
  classname = m.__class__.__name__
  # for every Linear layer in a model..
  if classname.find('Linear') != -1:
    # get the number of the inputs
    n = m.in_features
    y = 1.0/np.sqrt(n)
    m.weight.data.uniform_(-y, y)
    m.bias.data.fill_(0)


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
  embedding = torch.zeros(latitude.size(0), FOV_EMB_SIZE).cuda()

  embedding[:,  0:quarter*1] = torch.sin(latitude)
  embedding[:, quarter*1:quarter*2] = torch.cos(latitude)
  embedding[:, quarter*2:quarter*3] = torch.sin(longitude)
  embedding[:, quarter*3:quarter*4] = torch.cos(longitude)

  return embedding


def get_objects_classes(obj_dict_file,
                        data_path='../py_bottom_up_attention/demo/data/genome/1600-400-20'):

  vg_classes = []
  with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
      vg_classes.append(object.split(',')[0].lower().strip())

  vg2idx = json.load(
      open(obj_dict_file, 'r'))['vg2idx']
  idx2vg = json.load(
      open(obj_dict_file, 'r'))['idx2vg']

  obj_classes = ['']*len(idx2vg)
  for idx in idx2vg:
    obj_classes[int(idx)] = vg_classes[idx2vg[idx]]
  return vg2idx, idx2vg, obj_classes


def get_objects(move_x, move_y, nodes, vg2idx,
                full_w=4552,
                full_h=2276,
                fov_size=400):

  n_objects = len(vg2idx)

  all_regions = {}
  all_obj_list = {}
  for dir_method in DIRECTIONS:
    directions = DIRECTIONS[dir_method]
    regions = {d: np.zeros((n_objects, )) for d in directions}
    obj_list = {d: [] for d in directions}

    all_regions[dir_method] = regions
    all_obj_list[dir_method] = obj_list

  for node in nodes:
    x = nodes[node]['x']
    y = nodes[node]['y']
    vg_obj_id = str(nodes[node]['obj_id'])

    if vg_obj_id not in vg2idx:
      continue
    if move_x < x:
      d_left = move_x + full_w - x
      d_right = x - move_x
    else:
      d_left = move_x - x
      d_right = x + full_w - move_x
    if move_y < y:
      d_up = full_h
      d_down = y - move_y
    else:
      d_up = move_y - y
      d_down = full_h
    for d, dist in zip(DIRECTIONS['canonical'], [d_up, d_down, d_left, d_right]):
      if fov_size/2 <= dist <= fov_size*1.5 and vg_obj_id in vg2idx:

        obj_id = vg2idx[vg_obj_id]
        all_regions['canonical'][d][obj_id] = 1
        all_obj_list['canonical'][d].append(obj_id)

    for d, distances in zip(DIRECTIONS['cartesian'], [[d_up, d_down], [d_left, d_right]]):
      if (fov_size/2 <= distances[0] <= fov_size*1.5 or fov_size/2 <= distances[1] <= fov_size*1.5) and vg_obj_id in vg2idx:
        obj_id = vg2idx[vg_obj_id]
        all_regions['cartesian'][d][obj_id] = 1
        all_obj_list['cartesian'][d].append(obj_id)

    for d, distances in zip(DIRECTIONS['lup'], [[d_left, d_right], [d_up, d_up], [d_down, d_down]]):
      if (fov_size/2 <= distances[0] <= fov_size*1.5 or fov_size/2 <= distances[1] <= fov_size*1.5) and vg_obj_id in vg2idx:
        obj_id = vg2idx[vg_obj_id]
        all_regions['lup'][d][obj_id] = 1
        all_obj_list['lup'][d].append(obj_id)

    for d, dist, (low, high) in zip(DIRECTIONS['canonical_proximity'], [d_up, d_down, d_left, d_right, d_up, d_down, d_left, d_right], [(0.5, 1.5), (0.5, 1.5), (0.5, 1.5), (0.5, 1.5), (1.5, 2.5), (1.5, 2.5), (1.5, 2.5), (1.5, 2.5)]):
      if fov_size * low <= dist <= fov_size * high and vg_obj_id in vg2idx:

        obj_id = vg2idx[vg_obj_id]
        all_regions['canonical_proximity'][d][obj_id] = 1
        all_obj_list['canonical_proximity'][d].append(obj_id)

  return all_regions, all_obj_list


def rad2degree(lat, lng,
               adjust=False):
  '''Convert radians to degrees.
  '''

  adj_lat = 0
  adj_lng = 0
  if adjust:
    adj_lat = -0.015
    adj_lng = 0.1

  lng = np.degrees(lng + adj_lng)
  lat = np.degrees(lat + adj_lat)

#  lat = (lat + 180) % 360
  if lat > 180:
    lat = lat - 360
  return lat, lng


def get_det2_features(detections):
  boxes = []
  obj_classes = []

  for box, obj_class in zip(detections.pred_boxes, detections.pred_classes):
    box = box.cpu().detach().numpy()
    boxes.append(box)
    obj_class = obj_class.cpu().detach().numpy().tolist()
    obj_classes.append(obj_class)
  return boxes, obj_classes


def generate_grid(full_w=4552,
                  full_h=2276,
                  degree=30):
  left_w = int(full_w * (degree/360)+1)

  dx = full_w * (degree/360)
  dy = full_h * (degree/180)
  DISTANCE = (dx ** 2 + dy ** 2) ** 0.5 + 10

  size = 10

  objects = []
  nodes = []

  for lng in range(-75, 75, degree):
    for lat in range(0, 360, degree):
      gt_x = int(full_w * ((lat)/360.0))
      gt_y = int(full_h - full_h * ((lng + 90)/180.0))

      objects.append((lat, lng, 2, gt_x, gt_y, []))
      nodes.append([gt_x, gt_y])

  canvas = np.zeros((full_h, full_w, 3), dtype='uint8')

  node_dict = dict()
  for kk, o in enumerate(objects):
    o_type, ox, oy = o[2], o[3], o[4]
    o_label = '<START>'
    if o_type > 0:
      o_label = ''

    #cv2.putText(canvas, o_label, (ox+size, oy+size), font, 3, clr, 5)
    n = {
        'id': kk,
        'lat': o[0],
        'lng': o[1],
        'obj_label': o_label,
        'obj_id': o_type,
        'x': o[3],
        'y': o[4],
        'boxes': o[5],
        'neighbors': []
    }
    node_dict[kk] = n

  color = (125, 125, 125)

  n_nodes = len(nodes)
  order2nid = {i: i for i in range(n_nodes)}

  idx = n_nodes
  new_nodes = nodes
  for ii, n in enumerate(nodes):
    if n[0] < left_w:
      order2nid[idx] = ii
      new_nodes.append((n[0]+full_w, n[1]))
      idx += 1

  for ii, s1 in enumerate(new_nodes):
    for jj, s2 in enumerate(new_nodes):
      if ii == jj:
        continue

      d = ((s1[0]-s2[0])**2 + (s1[1]-s2[1])**2)**0.5
      if d <= DISTANCE:

        n0 = order2nid[ii]
        n1 = order2nid[jj]

        node_dict[n0]['neighbors'] += [n1]
        node_dict[n1]['neighbors'] += [n0]

        cv2.line(canvas, (s1[0], s1[1]),
                 (s2[0], s2[1]), color, 3, 8)
  for kk, o in enumerate(objects):
    o_type, ox, oy = o[2], o[3], o[4]

    canvas[oy-size:oy+size, ox-size:ox+size, 0] = 255.
    canvas[oy-size:oy+size, ox-size:ox+size, 1:] = 0

  return node_dict, canvas


def generate_gt_moves(image_path, move_list, move_ids,
                      fov_root='',
                      fov_size=400):

  cam = camera(output_image_shape=(fov_size, fov_size))
  cam.load_img(image_path)
  for jj, (move_id, fovs) in enumerate(zip(move_ids, move_list)):
    for kk, fov in enumerate(fovs):
      lat, lng = fov[0], fov[1]
      cam.look(lat, lng)

      fov_img = cam.get_image()
      fov_prefix = os.path.join(fov_root, '{}.gt_move.'.format(move_id))
      fov_file = fov_prefix + 'move{}.jpg'.format(kk)
      cv2.imwrite(fov_file, fov_img)


def generate_fovs(image_path, node_path, fov_prefix,
                  full_w=4552,
                  full_h=2276,
                  fov_size=400):

  cam = camera(output_image_shape=(fov_size, fov_size))
  cam.load_img(image_path)
  nodes = np.load(node_path,
                  allow_pickle=True)[()]

  for jj, n in enumerate(nodes.keys()):
    idx, lat, lng = nodes[n]['id'], nodes[n]['lat'], nodes[n]['lng']
    cam.look(lat, lng)

    fov = cam.get_image()
    cv2.imwrite(fov_prefix + '{}.jpg'.format(idx), fov)


def get_graph_hops(nodes, actions, image_path,
                   full_w=4552,
                   full_h=2276,
                   fov_size=512):
  '''Generate ground-truth paths for graph-based navigation.
  '''

  all_fovs = []

  for ii, inst in enumerate(actions[-1]['action_list'].split('_')):
    kk = 0
    prev_lat, prev_lng = -1, -1
    fovs = []
    for jj, fov in enumerate(inst.split('|')[1:]):
      rlat, rlng = float(fov.split(',')[0]), float(fov.split(',')[1])
      if prev_lat == rlat and prev_lng == rlng:
        continue

      prev_lat, prev_lng = rlat, rlng
      lat, lng = rad2degree(rlat, rlng, adjust=True)
      fx = int(full_w * ((lat + 180)/360.0))
      fy = int(full_h - full_h *
               ((lng + 90)/180.0))

      fovs.append((lat, lng, '{}.{}'.format(ii, kk), fx, fy))
      kk += 1
    all_fovs.append(fovs)

  gt_hops = []

  for fovs in all_fovs:

    closest_distances = []
    closest_nodes = []
    for fov in fovs:
      closest_d = full_h + full_w

      _, _, _, fx, fy = fov[0], fov[1], fov[2], fov[3], fov[4]

      for jj, n in enumerate(nodes.keys()):
        ox, oy = nodes[n]['x'], nodes[n]['y']

        d = ((fx-ox)**2 + (fy-oy)**2)**0.5
        if d < closest_d:
          closest_d = d
          closest = jj
      closest_distances.append(closest_d)
      closest_nodes.append(closest)
    gt_hops.append(closest_nodes)

  fov_dict = dict()

  kk = 0
  for sid, fovs in enumerate(all_fovs):
    for hid, fov in enumerate(fovs):
      n = gt_hops[sid][hid]

      f = {
          'lat': fov[0],
          'lng': fov[1],
          'label': fov[2],
          'x': fov[3],
          'y': fov[4],
          'closest_node': n,
          'sentence_id': sid,
          'hop_id': hid,
      }
      fov_dict[kk] = f
      kk += 1

  new_nodes = dict()
  for jj, n in enumerate(nodes.keys()):
    node = nodes[n]
    new_nodes[n] = node
    new_nodes[n]['neighbors'] = sorted([n]+list(set(node['neighbors'])))
  return fov_dict, gt_hops, new_nodes


def get_moves(instance, gt_lat, gt_lng, n_sentences,
              verbose=False):
  '''Return ground-truth lat/lng.
  '''

  all_moves = []
  ids = []
  flag = False

  for action in instance['actions']:
    if len(action['action_list'].split('_')) != n_sentences:
      flag = True
      break
    gt_moves = []
    for moves in action['action_list'].split('_'):
      move = moves.split('|')[-1]

      latitude, longitude = rad2degree(
          float(move.split(',')[0]), float(move.split(',')[1]), adjust=True)

      gt_moves.append((latitude, longitude))

    dist = ((gt_moves[-1][0] - gt_lat)**2 +
            (gt_moves[-1][1] - gt_lng)**2) ** 0.5
    if dist < 1:
      continue
    else:
      all_moves.append(gt_moves)
      ids.append(action['actionid'])

  if verbose:
    for action in instance['actions']:
      print(action['action_list'].split('_'))
      print('.')
    print('_')
    for moves in all_moves:
      print(moves)
      print('.')
  return flag, all_moves, ids


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
                          for x in range(width)] for y in range(height)]).float().cuda()
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


def load_datasets(splits, image_categories='all',
                  root='../data/dumps'):
  '''Loads a dataset dump.
  '''
  d, s = [], []
  for split_name in splits:
    fname = os.path.join(
        root, '{}.[{}].imdb.npy'.format(split_name, image_categories))
    print('loading split from {}'.format(fname))
    dump = np.load(fname, allow_pickle=True)[()]
    d += dump['data_list']
    s += dump['sentences']

  return d, s


def dump_datasets(splits, image_categories, output_file,
                  task='continuous_grounding',
                  task_root='',
                  graph_root='',
                  full_w=4552,
                  full_h=2276,
                  obj_dict_file='../data/vg_object_dictionaries.all.json',
                  degree=30):
  '''Prepares and dumps dataset to an npy file.
  '''

  if task_root != '' and not os.path.exists(task_root):
    try:
      os.makedirs(task_root)
    except:
      print('Cannot create folder {}'.format(task_root))
      quit(1)
  vg2idx = json.load(open(obj_dict_file, 'r'))['vg2idx']

  banned_turkers = set(['onboarding', 'vcirik'])
  data_list = []

  all_sentences = []

  image_set = set(CAT2LOC.keys())
  if image_categories != 'all':
    image_set = set(image_categories.split(','))
    for image in image_set:
      if image not in CAT2LOC:
        raise NotImplementedError(
            'Image Category {} is not in the dataset'.format(image))
  stats = defaultdict(int)
  for split in splits:
    data = []
    instances = json.load(open('../data/{}.json'.format(split), 'r'))

    pbar = tqdm(instances)
    for ii, instance in enumerate(pbar):
      if instance['turkerid'] not in banned_turkers:
        if 'ann_cat' in instance and instance['ann_cat'] == 'M' and split == 'train':
          continue
        datum = {}

        latitude, longitude = rad2degree(instance['latitude'],
                                         instance['longitude'],
                                         adjust=True)

        datum['annotationid'] = instance['annotationid']
        datum["gt_lng"] = longitude
        datum["gt_lat"] = latitude

        img_category = '_'.join(instance['imageurl'].split(
            '/')[-1].split('.')[0].split('_')[2:])
        pano_name = "_".join(instance['imageurl'].split(
            '/')[-1].split('.')[0].split('_')[:2]) + ".jpg"

        if img_category not in CAT2LOC or img_category not in image_set:
          continue
        img_loc = CAT2LOC[img_category]
        datum['pano'] = '../data/refer360images/{}/{}/{}'.format(
            img_loc, img_category, pano_name)
        datum['img_category'] = img_category
        stats[img_loc] += 1
        stats[img_category] += 1
        sentences = []
        sent_queue = []
        for refexp in instance['refexp'].replace('\n', '').split('|||')[1:]:
          if refexp[-1] not in ['.', '!', '?', '\'']:
            refexp += '.'
          sentences.append(refexp.split())
          sent_queue += [refexp]

        err, all_moves, move_ids = get_moves(
            instance, latitude, longitude, len(sentences))

        if err or all_moves == [] or len(all_moves[0]) != len(sentences):
          continue

        pano = "_".join(instance['imageurl'].split(
            '/')[-1].split('.')[0].split('_')[:2])

        datum['gt_moves'] = all_moves
        datum['refexps'] = sentences
        all_sentences += sent_queue

        if task == 'continuous_grounding':
          data.append(datum)
        if task == 'graph_grounding' and task_root != '':
          node_path = os.path.join(task_root, '{}.npy'.format(pano))
          node_img = os.path.join(task_root, '{}.jpg'.format(pano))
          fov_prefix = os.path.join(task_root, '{}.fov'.format(pano))
          nodes = np.load(node_path, allow_pickle=True)[()]
          fovs, graph_hops, new_nodes = get_graph_hops(nodes,
                                                       instance['actions'],
                                                       datum['pano'])
          assert len(sentences) == len(graph_hops)
          datum['fovs'] = fovs
          datum['graph_hops'] = graph_hops
          datum['nodes'] = new_nodes
          datum['node_img'] = node_img
          datum['fov_prefix'] = fov_prefix
          data.append(datum)
        elif task == 'fov_pretraining' and task_root != '':
          node_path = os.path.join(graph_root, '{}.npy'.format(pano))
          node_img = os.path.join(graph_root, '{}.jpg'.format(pano))
          fov_prefix = os.path.join(graph_root, '{}.fov'.format(pano))
          nodes = np.load(node_path, allow_pickle=True)[()]
          fovs, graph_hops, new_nodes = get_graph_hops(nodes,
                                                       instance['actions'],
                                                       datum['pano'])

          for moves, move_id in zip(all_moves, move_ids):
            for mm, move in enumerate(moves):
              mdatum = {}
              mdatum['move_id'] = mm
              mdatum['move_max'] = len(sentences)
              mdatum['pano'] = datum['pano']
              mdatum['actionid'] = move_id
              mdatum['annotationid'] = instance['annotationid']

              lat, lng = move[0], move[1]
              mx = int(full_w * ((lat + 180)/360.0))
              my = int(full_h - full_h *
                       ((lng + 90)/180.0))

              mdatum['latitude'] = move[0]
              mdatum['longitude'] = move[1]
              mdatum['x'] = mx
              mdatum['y'] = my

              mdatum['refexp'] = sentences[mm]
              fov_prefix = os.path.join(
                  task_root, '{}.gt_move.'.format(move_id))
              fov_file = fov_prefix + 'move{}.jpg'.format(mm)
              mdatum['fov_file'] = fov_file
              regions, obj_list = get_objects(mx, my, nodes, vg2idx)
              mdatum['regions'] = regions
              mdatum['obj_list'] = obj_list

              directions = [len(obj_list['canonical'][d]) > 0 for d in [
                  'up', 'down', 'left', 'right']]

              if any(directions):
                data.append(mdatum)
        elif task == 'fov_pretraining_grid' and task_root != '':
          node_path = os.path.join(graph_root, '{}.npy'.format(pano))
          node_img = os.path.join(graph_root, '{}.jpg'.format(pano))
          fov_prefix = os.path.join(graph_root, '{}.fov'.format(pano))
          nodes = np.load(node_path, allow_pickle=True)[()]
          fovs, graph_hops, new_nodes = get_graph_hops(nodes,
                                                       instance['actions'],
                                                       datum['pano'])
          grid_nodes, _ = generate_grid(degree=degree)

          for n in grid_nodes:
            node = grid_nodes[n]

            mdatum = {}
            fov_id = node['id']
            mdatum['fov_id'] = fov_id

            mdatum['move_max'] = len(sentences)
            mdatum['pano'] = datum['pano']
            # mdatum['actionid'] = move_id
            mdatum['annotationid'] = instance['annotationid']

            lat, lng = node['lat'], node['lng']
            mx, my = node['x'], node['y']
            mdatum['latitude'] = lat
            mdatum['longitude'] = lng
            mdatum['x'] = mx
            mdatum['y'] = my

            mdatum['refexps'] = sentences
            fov_file = os.path.join(
                task_root, '{}.fov{}.jpg'.format(pano, fov_id))

            mdatum['fov_file'] = fov_file
            regions, obj_list = get_objects(mx, my, nodes, vg2idx)
            mdatum['regions'] = regions
            mdatum['obj_list'] = obj_list

            directions = [len(obj_list['canonical'][d]) > 0 for d in [
                'up', 'down', 'left', 'right']]

            if any(directions):
              data.append(mdatum)
    data_list.append(data)
    pbar.close()

  print('_'*20)
  for c in stats:
    print('{:<16} {:2.2f}'.format(c, stats[c]))
  print('_'*20)

  print('Dumping to {}'.format(output_file))
  np.save(open(output_file, 'wb'), {'data_list': data_list,
                                    'sentences': all_sentences})
#  return data_list, all_sentences


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
    return seq_tensor.cuda(), mask.cuda(), emb_mask.cuda(), seq_lengths.cuda()
  return seq_tensor, mask, emb_mask, seq_lengths


def add_overlay(src, trg, coor,
                maxs_x=400, maxs_y=400,
                maxt_x=60, maxt_y=60,
                ignore=[76, 112, 71]):
  '''Adds an overlay of waldo icon for oracle experiments.
  '''
  xs_min = max(0, coor[1] - int(maxt_x / 2))

  xs_max = min(maxs_x, coor[1] + int(maxt_x / 2))
  ys_min = max(0, coor[0] - int(maxt_y / 2))
  ys_max = min(maxs_y, coor[0] + int(maxt_y / 2))

  xt_min = max(0, int(maxt_x / 2) - coor[1])
  xt_max = maxt_x - min(coor[1] + int(maxt_x / 2) - maxs_x, 0)
  yt_min = max(0, int(maxt_y / 2) - coor[0])
  yt_max = maxt_y - max(coor[0] + int(maxt_y / 2) - maxs_y, 0)

  if ignore == []:
    src[ys_min:ys_max,
        xs_min:xs_max] = trg[yt_min:yt_max,
                             xt_min:xt_max]

  else:
    for xs, xt in zip(range(xs_min, xs_max), range(xt_min, xt_max)):
      for ys, yt in zip(range(ys_min, ys_max), range(yt_min, yt_max)):
        pixel = trg[yt, xt, :].tolist()
        if pixel != ignore:
          src[ys, xs, :] = trg[yt, xt, :]
  return src


from models.finder import Finder


def get_model(args, vocab, n_actions=5):
  '''Returns a Finder model.
  '''
  if args.model == 'visualbert':
    args.use_masks = True
    args.cnn_layer = 2
    args.n_emb = 512

  return Finder(args, vocab, n_actions).cuda()


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


def compute_precision_with_logits(logits, labels_vector,
                                  precision_k=1,
                                  mask=None):
  labels = torch.zeros(*logits.size()).cuda()
  for ll in range(logits.size(0)):
    labels[ll, :].scatter_(0, labels_vector[ll], 1)

  adjust_score = False
  if type(mask) != type(None):
    labels = labels * mask.unsqueeze(0).expand(labels.size())
    adjust_score = True
  one_hots = torch.zeros(*labels.size()).cuda()
  logits = torch.sort(logits, 1, descending=True)[1]
  one_hots.scatter_(1, logits[:, :precision_k], 1)

  batch_size = logits.size(0)
  scores = ((one_hots * labels).sum(1) >= 1).float().sum() / batch_size
  if adjust_score:
    valid_rows = (labels.sum(1) > 0).sum(0)
    if valid_rows == 0:
      scores = torch.tensor(1.0).cuda()
    else:
      hit = (scores * batch_size)
      scores = hit / valid_rows
  return scores
