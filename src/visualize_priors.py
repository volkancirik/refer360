from box_utils import calculate_iou
import base64
import paths
from collections import defaultdict
import os
import numpy as np
from utils import add_overlay
from utils import get_coordinates
from utils import get_nearest
from utils import visualize_path
from utils import add_square
from utils import color_red, color_green, color_blue
from utils import SPLITS
from utils import CAT2LOC
from utils import get_object_dictionaries
import json
import panoramic_camera_cached as camera
import random
import sys
import cv2
import matplotlib
from pprint import pprint
import argparse
import csv
matplotlib.use('Agg')
csv.field_size_limit(sys.maxsize)


def get_prior_predictions(boxes, object_ids, cooccurrence, vg2name,
                          threshold=0.25):

  dir2obj = defaultdict(list)

  for ii, box in enumerate(boxes):
    directions = []
    # box[1] < 120:  # 200:
    if calculate_iou(box, [0, 0, 400, 120]) > 0.1:
      directions.append('u')
      # box[1] > 280:  # 200:
    elif calculate_iou(box, [0, 280, 400, 400]) > 0.1:
      directions.append('d')
    # box[0] < 120:  # 200:
    if calculate_iou(box, [0, 0, 120, 400]) > 0.1:
      if directions:
        directions[0] += 'l'
      directions.append('l')
    # box[0] > 280:  # 200:
    elif calculate_iou(box, [280, 0, 400, 400]) > 0.1:
      if directions:
        directions[0] += 'r'
      directions.append('r')
    for direction in directions:
      dir2obj[direction] += [ii]

  dir_probs = {direction: defaultdict(float) for direction in dir2obj.keys()}
  for direction in dir2obj.keys():
    indexes = dir2obj[direction]
    #feat_index = DIR2IDX[direction]

    # for each object on the edge
    for index in indexes:
      o = object_ids[index]
      name = vg2name.get(o, '</s>')
      idx = obj_classes.index(name)

      for co_idx in range(cooccurrence.shape[1]):
        co_name = obj_classes[co_idx]
        if cooccurrence[idx, co_idx] > 0:
          dir_probs[direction][co_name] += cooccurrence[idx, co_idx]

  for direction in dir_probs.keys():
    del_list = []
    for obj in dir_probs[direction].keys():
      if dir_probs[direction][obj] < threshold:
        del_list.append(obj)
    for obj in del_list:
      del dir_probs[direction][obj]
  return dir_probs


def load_butd(butd_filename,
              threshold=0.5,
              w2v=None,
              vg2name=None,
              keys=['features']):
  fov2key = {k: {} for k in keys}

  FIELDNAMES = ['img_id', 'img_h', 'img_w', 'objects_id', 'objects_conf',
                'attrs_id', 'attrs_conf', 'num_boxes', 'boxes', 'features']

  with open(butd_filename) as f:
    reader = csv.DictReader(f, FIELDNAMES, delimiter='\t')
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
        try:
          item[key] = item[key].reshape(shape)
        except:
          if key == 'boxes':
            dim = 4
          elif key == 'features':
            dim = 2048
          else:
            dim = 1
          item[key] = np.zeros((boxes, dim))
          pass
        item[key].setflags(write=False)

      keep_boxes = np.where(item['objects_conf'] >= threshold)[0]
      obj_ids = item['objects_id'][keep_boxes]

      scanId, viewpointId = item['img_id'].split('.')
      pano_fov = '{}_{}'.format(scanId, viewpointId)
      for k in keys:
        if k not in item:
          print('{} not in BUTD file'.format(k))
          quit(1)
        if k == 'features':
          feats = item['features'][keep_boxes]
          if w2v != None and vg2name != None:
            emb_feats = np.zeros((feats.shape[0], 300), dtype=np.float32)
            for ii, obj_id in enumerate(obj_ids):
              obj_name = vg2name.get(obj_id, '</s>')
              emb_feats[ii, :] = w2v.get(obj_name, w2v['</s>'])
            feats = emb_feats
          feats = np.sum(feats, axis=0)
          fov2key[k][pano_fov] = feats
        else:
          fov2key[k][pano_fov] = item[k][keep_boxes]
  return fov2key


def randomly_pick_datum(act_instances, categories):
  while True:
    actionid = random.choice(list(act_instances.keys()))
    datum = act_instances[actionid]
    image_path = datum['img_src']
    category = image_path.split('/')[4]
    if category in categories:
      return actionid


def load_datum(datum, cam):

  refer = "\n".join([" ".join(refexp) for refexp in datum['refexp']])
  # pprint(datum)
  n_steps = len(datum['act_deg_list'])
  image_path = datum['img_src']

  waldo_position_lng = datum['gt_longitude']
  waldo_position_lat = datum['gt_latitude']

  pred_lng = datum['pred_xlng_deg']
  pred_lat = datum['pred_ylat_deg']

  gt_x, gt_y = get_coordinates(waldo_position_lng,
                               waldo_position_lat)
  pred_x, pred_y = get_coordinates(pred_lng,
                                   pred_lat)

  print("image_path", image_path)
  print(refer)
  print("gt:", waldo_position_lat, waldo_position_lng)
  print("gt xy:", gt_x, gt_y)
  print("pred:", pred_lat, pred_lng)
  print("pred xy:", pred_x, pred_y)
  print('n_steps:', n_steps)

  panels = {}
  panels['original'] = cv2.imread(
      image_path, cv2.IMREAD_COLOR)
  panels['grid'] = cv2.imread(
      grid_path, cv2.IMREAD_COLOR)
  panels['mask'] = np.zeros((full_h, full_w, 3), dtype='uint8')

  # add_square(panels['original'], gt_x, gt_y, color=color_green)
  # add_square(panels['original'], pred_x, pred_y, color=color_red)
  # add_square(panels['grid'], gt_x, gt_y, color=color_green)
  # add_square(panels['grid'], pred_x, pred_y, color=color_red)

  start_loc = datum['act_deg_list'][0][0]
  start_x, start_y = get_coordinates(start_loc[0], start_loc[1],
                                     full_w=full_w,
                                     full_h=full_h)
  start_fov, _ = get_nearest(nodes, start_x, start_y)
  gt_path = [start_fov]
  for kk, act_list in enumerate(datum['act_deg_list']):
    act = act_list[-1]
    lng, lat = act
    x, y = get_coordinates(lng, lat,
                           full_w=full_w,
                           full_h=full_h)

    min_n, _ = get_nearest(nodes, x, y)
    if gt_path[-1] != min_n:
      gt_path.append(min_n)
    if len(gt_path) >= 2:
      start = gt_path[-2]
      end = gt_path[-1]
      intermediate_path = cam.paths[start][end]
      panels['grid'] = visualize_path(
          intermediate_path, cam.nodes, cam.edges, panels['grid'])
      print('intermediate_path:', intermediate_path)
  print('gt_path:', gt_path)
  return image_path, gt_path, panels, waldo_position_lng, waldo_position_lat, pred_lng, pred_lat


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--threshold', type=float,  default=0.25,
                      help='threshold for object detection, default=0.25')
  parser.add_argument('--degree', type=int,  default=60,
                      help='degrees between fovs, default=60')
  parser.add_argument('--full_w', type=int,  default=4552,
                      help='full width of the pano, default=4552')
  parser.add_argument('--full_h', type=int,  default=2276,
                      help='full width of the pano, default=2276')
  parser.add_argument('--box_size', type=int,  default=20,
                      help='box size for fov centers, default=20')
  parser.add_argument('--cache_root', type=str,
                      default='../data/r360tiny_data/cached_data_60degrees',
                      help='cache_root, default="../data/r360tiny_data/cached_data_60degrees"')
  parser.add_argument('--data_root', type=str,
                      default='../data/r360tiny_data',
                      help='data root, default="../data/r360tiny_data"')
  parser.add_argument('--dump_root', type=str,
                      default='visualize_out',
                      help='dump root, default="visualize_r360tiny_out/"')
  parser.add_argument('--butd_filename', type=str,
                      default='../data/r360tiny_data/img_features/r360tiny_60degrees_obj36.tsv',
                      help='object detection file, default="../data/r360tiny_data/img_features/r360tiny_60degrees_obj36.tsv"')
  parser.add_argument('--cooccurrence_file', type=str,
                      default='../data/cooccurrences/cooccurrence.diagonal.npy',
                      help='object detection file, default="../data/cooccurrences/cooccurrence.diagonal.npy"')
  parser.add_argument('--obj_dict_file', type=str,
                      default='../data/vg_object_dictionaries.all.json',
                      help='object detection file, default="../data/vg_object_dictionaries.all.json"')
  parser.add_argument("--images",
                      type=str,
                      default='all',
                      help='list of image categories comma separated or all')
  parser.add_argument('--actionid', type=int,  default=-1,
                      help='actionid for instance, default=-1 (random)')

  args = parser.parse_args()
  pprint(args)

  font = cv2.FONT_HERSHEY_SIMPLEX
  full_w, full_h, size = args.full_w, args.full_h, args.box_size
  cache_root, data_root, dump_root = args.cache_root, args.data_root, args.dump_root
  butd_filename, obj_dict_file = args.butd_filename, args.obj_dict_file
  image_categories = args.images
  actionid = args.actionid

  grid_path = os.path.join(cache_root, 'canvas.jpg')
  meta_file = os.path.join(cache_root, 'meta.npy')
  meta = np.load(meta_file, allow_pickle=True)[()]
  nodes = meta['nodes']
  waldo_img = cv2.resize(cv2.imread(
      '../data/waldo.png', cv2.IMREAD_COLOR), (60, 40), interpolation=cv2.INTER_AREA)
  target_img = cv2.resize(cv2.imread(
      '../data/target.png', cv2.IMREAD_COLOR), (60, 60), interpolation=cv2.INTER_AREA)
  print('loading BUTD boxes...', butd_filename)

  vg2idx, idx2vg, obj_classes, name2vg, name2idx, vg2name = get_object_dictionaries(
      obj_dict_file, return_all=True)
  fov2keys = load_butd(butd_filename,
                       vg2name=vg2name,
                       keys=['boxes', 'objects_id'])
  cooccurrence_data = np.load(args.cooccurrence_file,
                              allow_pickle=True)[()]
  cooccurrence = cooccurrence_data['cooccurrence']
  normalize_column = 'prompt' in cooccurrence_data['method']
  if normalize_column:
    print('will normalize columns')
    for idx in range(cooccurrence.shape[0]):
      sum_count = np.sum(cooccurrence[:, idx])
      if sum_count > 0:
        cooccurrence[:, idx] = cooccurrence[:, idx] / sum_count

  for idx in range(cooccurrence.shape[0]):
    sum_count = np.sum(cooccurrence[idx, :])
    if sum_count > 0:
      cooccurrence[idx, :] = cooccurrence[idx, :] / sum_count

  data_list = [json.load(open(data_root+'/{}.json'.format(split)))
               for split in SPLITS]
  act_instances = {}
  pano2act = defaultdict(list)
  pano2ann = defaultdict(list)

  for data in data_list:
    for instance in data:
      for action in instance['actions']:
        action['img_src'] = instance['img_src']
        action['refexp'] = instance['refexp']
        action['gt_longitude'] = instance['xlng_deg']
        action['gt_latitude'] = instance['ylat_deg']
        image_path = instance['img_src']
        pano_id = image_path.split('/')[5].split('.')[0]

        act_instances[action['actionid']] = action

        pano2act[pano_id].append(action['actionid'])
        pano2ann[pano_id].append(action['annotationid'])

  if dump_root != '' and not os.path.exists(dump_root):
    try:
      os.makedirs(dump_root)
    except:
      print('Cannot create folder {}'.format(dump_root))

  if image_categories == 'all':
    categories = set(CAT2LOC.keys())
  else:
    categories = set()
    for cat in image_categories:
      if cat not in CAT2LOC:
        print('{} not a part of image categories'.format(cat))
        quit(1)
      categories.add(cat)
  print('image categories:', categories)
  if actionid >= 0:
    pass
  else:
    #    actionid = random.choice(list(act_instances.keys()))
    actionid = randomly_pick_datum(act_instances, categories)
    print('randomly picked', actionid)

  demo_list = [actionid]
  demo_id = 0
  cam = camera.CachedPanoramicCamera(cache_root)
  cam.load_maps()
  cam.load_masks()

  load_new = True
  while True:

    if load_new:
      load_new = False
      fov_id = 0
      datum = act_instances[actionid]
      image_path, gt_path, panels, waldo_position_lng, waldo_position_lat, pred_lng, pred_lat = load_datum(
          datum, cam)
      cam.load_img(image_path)
      cam.load_fovs()
      n_steps = len(gt_path)

    image_path = datum['img_src']
    pano_id = image_path.split('/')[5].split('.')[0]
    print('pano_id:', pano_id)
    print('actions:', pano2act[pano_id])

    cam.look_fov(gt_path[fov_id])
    fov_key = '{}_{}'.format(pano_id, gt_path[fov_id])
    try:
      boxes, obj_ids = fov2keys['boxes'][fov_key], fov2keys['objects_id'][fov_key]
    except:
      boxes, obj_ids = np.array([]), np.array([])
      pass

    dir_probs = get_prior_predictions(boxes, obj_ids, cooccurrence, vg2name,
                                      args.threshold)
    pprint(dir_probs)

    img = cam.get_image()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    curr_x, curr_y = cam.get_current()
    lng_map, lat_map = cam.get_pixel_map()
    mask = cam.get_mask()

    panels['mask'] = np.logical_or(panels['mask'], mask).astype(np.uint8)*255
    panels['semantic'] = cv2.bitwise_and(panels['original'], panels['mask'])

    waldo_coor = cam.get_image_coordinate_for(
        waldo_position_lng, waldo_position_lat)
    if waldo_coor is not None:
      img = add_overlay(img, waldo_img, waldo_coor, maxt_x=60,
                        maxt_y=40, ignore=[76, 112, 71])
    target_coor = cam.get_image_coordinate_for(pred_lng, pred_lat)
    if target_coor is not None:
      img = add_overlay(img, target_img, target_coor, ignore=[255, 255, 255])

    for obj_id in range(boxes.shape[0]):
      box = [int(v) for v in boxes[obj_id]]
      label = vg2name.get(obj_ids[obj_id], '</s>')
      cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color_red, 2)
      cv2.putText(img, label, (box[0], box[1] - 12),
                  font, 0.35, color_red, 1)

    grid = panels['grid'].copy()
    add_square(grid, curr_x, curr_y, color=color_blue)
    mask = cam.get_mask()
    grid = cv2.bitwise_and(grid, mask)

    original = cv2.resize(panels['original'], (1800, 900))
    grid = cv2.resize(grid, (1800, 900))
    mask = cv2.resize(panels['mask'], (1800, 900))
    semantic = cv2.resize(panels['semantic'], (1800, 900))
    for jj, refexp in enumerate(datum['refexp']):
      refer = " ".join(refexp)
      cv2.putText(original, refer, (5, (jj+1)*30),
                  font, 1.15, (0, 0, 255), 2)

    #resized_img = cv2.resize(img, (800, 800))
    peripheral = [np.zeros((600, 600, 3), dtype='uint8')]*9
    neighbors = cam.nodes[gt_path[fov_id]]['dir2neighbor']
    for kk, direction in enumerate(['ul', 'u', 'ur', 'l', 'c', 'r', 'dl', 'd', 'dr']):
      if direction in neighbors:
        peripheral[kk] = cv2.resize(cv2.cvtColor(
            cam.fovs[neighbors[direction]], cv2.COLOR_RGB2BGR), (600, 600))
        if direction not in dir_probs:
          continue
        for nn, obj in enumerate(dir_probs[direction]):
          obj_info = '{}:{:2.2f}'.format(obj, dir_probs[direction][obj])
          cv2.putText(peripheral[kk], obj_info, (5, (nn+1)*30),
                      font, 1.15, (0, 0, 255), 2)

    peripheral[4] = cv2.resize(img, (600, 600))

    row0 = np.concatenate(
        (peripheral[0], peripheral[1], peripheral[2]), axis=1)
    row1 = np.concatenate(
        (peripheral[3], peripheral[4], peripheral[5]), axis=1)
    row2 = np.concatenate(
        (peripheral[6], peripheral[7], peripheral[8]), axis=1)

    column0 = np.concatenate((row0, row1, row2), axis=0)
    column1 = np.concatenate((original, grid), axis=0)
    #column2 = np.concatenate((mask, semantic), axis=0)
    #combined = np.concatenate((column0, column1, column2), axis=1)
    combined = np.concatenate((column0, column1), axis=1)
    cv2.imshow(
        "Use left/right arrows to move FoV. (n)ext (p)revious example. (c)lose", combined)
    prefix = '{}.{}'.format(actionid, fov_id)
    for (d, f) in [(original, 'original'),
                   (grid, 'grid'),
                   (mask, 'mask'),
                   (semantic, 'semantic')]:
      fname = os.path.join(dump_root, prefix + '.' + f + '.png')
      cv2.imwrite(fname, d)

    # image_path = datum['img_src']
    # pano_id = image_path.split('/')[5]
    # print('pano_id:', pano_id)
    # print('actions:', pano2act[pano_id])

    key = cv2.waitKey()

    if key == 110:
      demo_id = demo_id + 1
      if demo_id == len(demo_list):
        print('randomly picked', actionid)
        # actionid = random.choice(list(act_instances.keys()))
        actionid = randomly_pick_datum(act_instances, categories)
        demo_list.append(actionid)
      else:
        actionid = demo_list[demo_id]
      load_new = True
    elif key == 112:
      demo_id = max(demo_id - 1, 0)
      actionid = demo_list[demo_id]
      load_new = True
    elif key == 2 or key == 98:
      if fov_id > 0:
        fov_id = fov_id - 1
    elif key == 3 or key == 102:
      if fov_id < n_steps-1:
        fov_id = (fov_id + 1)
    if key == 99:  # c
      break
# actions: [7851, 7805, 8160, 9952, 10309, 10659, 13339, 18016, 17797, 19009, 20677, 21314]
