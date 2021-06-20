"""Utils for io, tokenization etc.
"""
from collections import defaultdict
import cv2
import json
import numpy as np
import os
from tqdm import tqdm

from PIL import Image
from panoramic_camera_gpu import PanoramicCameraGPU as camera
# from panoramic_camera import PanoramicCamera as camera
from pprint import pprint

# dict of pano category to location category
CAT2LOC = {'restaurant': 'indoor',
           'shop': 'indoor',
           'expo_showroom': 'indoor',
           'living_room': 'indoor',
           'bedroom': 'indoor',
           'street': 'outdoor',
           'plaza_courtyard': 'outdoor',
           }
# list of dataset splits
SPLITS = [
    'validation.seen',
    'validation.unseen',
    'test.seen',
    'test.unseen',
    'train',
]
# dict direction types -> list of directions
DIRECTIONS = {
    'navigation': ['ul', 'u', 'ur', 'l', 'r', 'dl', 'd', 'dr'],
    'canonical': ['up', 'down', 'left', 'right'],
    'cartesian': ['vertical', 'horizontal'],
    'lup': ['lateral', 'up', 'down'],
    'canonical_proximity': ['close_up', 'close_down', 'close_left', 'close_right',
                            'far_up', 'far_down', 'far_left', 'far_right']
}
color_gray = (125, 125, 125)
color_dark = (25, 25, 25)

color_red = (255, 0, 0)
color_green = (0, 255, 0)
color_blue = (0, 0, 255)
color_yellow = (255, 255, 0)
color_white = (255, 255, 255)


def get_nearest(node_dict, x, y):
  '''returns the nearest fov to given x,y
  '''
  min_d = np.float('inf')
  for n in node_dict:
    d = ((x - node_dict[n]['x'])**2 + (y - node_dict[n]['y'])**2)**0.5
    if d < min_d:
      min_d = d
      min_n = n
  return min_n, min_d


def get_coordinates(xlng, ylat,
                    full_w=4552,
                    full_h=2276):
  '''given lng lat returns coordinates in panorama image
  '''
  x = int(full_w * ((xlng + 180)/360.0))
  y = int(full_h - full_h * ((ylat + 90)/180.0))
  return x, y


def get_object_dictionaries(obj_dict_file,
                            return_all=False):
  '''Loads object object dictionaries
  visual genome -> idx2
  idx ->  visual genome
  idx -> object classes'''

  print('loading object dictionaries from:', obj_dict_file)
  data = json.load(
      open(obj_dict_file, 'r'))
  vg2idx = data['vg2idx']
  idx2vg = data['idx2vg']
  obj_classes = data['obj_classes']

  if return_all:

    vg2idx = {int(k): int(vg2idx[k]) for k in vg2idx}
    idx2vg = {int(k): int(idx2vg[k]) for k in idx2vg}
    obj_classes.append('</s>')
    vg2idx[1601] = len(obj_classes)-1
    idx2vg[len(obj_classes)-1] = 1601

    name2vg, name2idx, vg2name = {}, {}, {}
    for idx in idx2vg:
      vg_idx = idx2vg[idx]
      obj_name = obj_classes[idx]

      name2vg[obj_name] = vg_idx
      name2idx[obj_name] = idx
      vg2name[vg_idx] = obj_name

    return vg2idx, idx2vg, obj_classes, name2vg, name2idx, vg2name
  return vg2idx, idx2vg, obj_classes


def coordinate2degrees(x, y,
                       full_w=4552,
                       full_h=2276):
  '''given x,y coordinates returrns lng lat degrees
  '''
  xlng = ((x / full_w) - 0.5) * 360.
  ylat = ((y / full_h) - 0.5) * 90
  return xlng, ylat


def objlist2regions(obj_list, n_objects,
                    dir_methods=[]):

  if dir_methods == []:
    dir_methods = DIRECTIONS.keys()

  all_regions = {}
  for dir_method in dir_methods:
    directions = DIRECTIONS[dir_method]
    regions = {d: np.zeros((n_objects, )) for d in directions}
    all_regions[dir_method] = regions

  for dir_method in dir_methods:
    for direction in obj_list[dir_method]:
      for obj in obj_list[dir_method][direction]:
        all_regions[dir_method][direction][obj] = 1.0
  return all_regions


def get_objects(move_x, move_y, nodes, vg2idx,
                full_w=4552,
                full_h=2276,
                fov_size=400,
                include_vectors=True):
  '''Given x,y and list of object returns dictionary for diffferent direction methods and a list of objects
  '''

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

    directions = []
    if fov_size/2 <= d_up <= fov_size*1.5:
      directions.append('u')
    elif fov_size/2 <= d_down <= fov_size*1.5:
      directions.append('d')
    if fov_size/2 <= d_left <= fov_size*1.5:
      if directions:
        directions[0] += 'l'
      directions.append('l')
    elif fov_size/2 <= d_right <= fov_size*1.5:
      if directions:
        directions[0] += 'r'
      directions.append('r')
    if vg_obj_id in vg2idx:
      for d in directions:
        obj_id = vg2idx[vg_obj_id]
        all_regions['navigation'][d][obj_id] = 1
        all_obj_list['navigation'][d].append(obj_id)

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

  if include_vectors:
    return all_regions, all_obj_list
  else:
    return {}, all_obj_list


def rad2degree(xlng, ylat,
               adjust=False):
  '''Convert radians to degrees.
  '''

  adj_lat = 0
  adj_lng = 0
  if adjust:
    adj_lng = -0.015
    adj_lat = 0.1

  xlng = np.degrees(xlng + adj_lng)
  ylat = np.degrees(ylat + adj_lat)

#  lat = (lat + 180) % 360
  if xlng > 180:
    xlng = xlng - 360
  return xlng, ylat


def generate_grid(full_w=4552,
                  full_h=2276,
                  degree=30):
  '''Generates grid of FoVs.
  '''
  left_w = int(full_w * (degree/360)+1)

  dx = full_w * (degree/360)
  dy = full_h * (degree/180)
  DISTANCE = (dx ** 2 + dy ** 2) ** 0.5 + 10

  size = 10

  objects = []
  nodes = []

  for ylat in range(-75, 75, degree):
    for xlng in range(0, 360, degree):
      gt_x = int(full_w * ((xlng)/360.0))
      gt_y = int(full_h - full_h * ((ylat + 90)/180.0))

      objects.append((xlng, ylat, 2, gt_x, gt_y, []))
      nodes.append([gt_x, gt_y])

  canvas = np.zeros((full_h, full_w, 3), dtype='uint8')

  node_dict = dict()
  for kk, o in enumerate(objects):
    o_type, ox, oy = o[2], o[3], o[4]
    o_label = '<START>'
    if o_type > 0:
      o_label = ''

    # cv2.putText(canvas, o_label, (ox+size, oy+size), font, 3, clr, 5)
    n = {
        'id': kk,
        'xlng': o[0],
        'ylat': o[1],
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
      xlng, ylat = fov[0], fov[1]
      cam.look(xlng, ylat)

      fov_img = cam.get_image()
      fov_prefix = os.path.join(fov_root, '{}.gt_move.'.format(move_id))
      fov_file = fov_prefix + 'move{}.jpg'.format(kk)
      # cv2.imwrite(fov_file, fov_img)
      pil_img = Image.fromarray(fov_img)
      pil_img.save(fov_file)


def generate_fovs(image_path, node_path, fov_prefix,
                  fov_size=400):
  '''Generates FoV images for given list of objects.
  '''

  cam = camera(output_image_shape=(fov_size, fov_size))
  cam.load_img(image_path)
  nodes = np.load(node_path,
                  allow_pickle=True)[()]

  for jj, n in enumerate(nodes.keys()):
    idx, xlng, ylat = nodes[n]['id'], nodes[n]['xlng'], nodes[n]['ylat']
    cam.look(xlng, ylat)

    fov_img = cam.get_image()
    fov_file = fov_prefix + '{}.jpg'.format(idx)
    # cv2.imwrite(fov_file, fov_img)
    pil_img = Image.fromarray(fov_img)
    pil_img.save(fov_file)


def get_graph_hops(nodes, actions, image_path,
                   full_w=4552,
                   full_h=2276,
                   fov_size=512):
  '''Generate ground-truth paths for graph-based navigation.
  '''

  all_fovs = []

  for ii, inst in enumerate(actions[-1]['act_deg_list'].split('_')):
    kk = 0
    prev_lat, prev_lng = -1, -1
    fovs = []
    for jj, fov in enumerate(inst.split('|')[1:]):
      xlng, ylat = fov
      if prev_lat == ylat and prev_lng == xlng:
        continue

      prev_lat, prev_lng = ylat, xlng
      fx = int(full_w * ((xlng + 180)/360.0))
      fy = int(full_h - full_h *
               ((ylat + 90)/180.0))

      fovs.append((xlng, ylat, '{}.{}'.format(ii, kk), fx, fy))
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
          'xlng': fov[0],
          'ylat': fov[1],
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


def get_moves(instance, gt_lng, gt_lat, sentences,
              verbose=False):
  '''Return ground-truth lat/lng.
  '''
  n_sentences = len(sentences)
  all_moves = []
  ids = []
  flag = True

  for action in instance['actions']:

    if len(action['act_deg_list']) != n_sentences:
      continue
    gt_moves = []
    for moves in action['act_deg_list']:
      move = moves[-1]
      xlongitude, ylatitude = move
      gt_moves.append((xlongitude, ylatitude))

    all_moves.append(gt_moves)
    ids.append(action['actionid'])
    flag = False

  if verbose:
    for action in instance['actions']:
      print(action['act_deg_list'].split('_'))
      print('.')
    print('_')
    for moves in all_moves:
      print(moves)
      print('.')
  return flag, all_moves, ids


def load_datasets(splits, image_categories='all',
                  root='../data/continuous_grounding'):
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
                  degree=30,
                  use_gt_moves=True,
                  cache_root='../data/cached_data_30degrees/',
                  data_root='../data'):
  '''Prepares and dumps dataset to an npy file.
  '''

  if task_root != '' and not os.path.exists(task_root):
    try:
      os.makedirs(task_root)
    except:
      print('Cannot create folder {}'.format(task_root))
      quit(1)

  if cache_root:
    meta_file = os.path.join(cache_root, 'meta.npy')
    meta = np.load(meta_file, allow_pickle=True)[()]
    cached_nodes = meta['nodes']
    cached_paths = meta['paths']
    add_cached_path = True
  else:
    add_cached_path = False

  vg2idx = json.load(open(obj_dict_file, 'r'))['vg2idx']

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
    instances = json.load(
        open(os.path.join(data_root, '{}.json'.format(split)), 'r'))

    pbar = tqdm(instances)
    count_err = 0
    for ii, instance in enumerate(pbar):
      if 'ann_cat' in instance and instance['ann_cat'] == 'M' and split == 'train':
        continue
      datum = {}

      xlongitude, ylatitude = instance['xlng_deg'], instance['ylat_deg'],

      datum['annotationid'] = instance['annotationid']
      gt_x, gt_y = get_coordinates(xlongitude, ylatitude)
      datum['gt_lng'] = xlongitude
      datum['gt_lat'] = ylatitude
      datum['gt_x'] = gt_x
      datum['gt_y'] = gt_y

      img_cat = instance['img_cat']
      img_loc = instance['img_loc']
      img_src = instance['img_src']
      if img_cat not in CAT2LOC or img_cat not in image_set:
        continue

      datum['img_src'] = img_src
      datum['img_category'] = img_cat
      stats[img_loc] += 1
      stats[img_cat] += 1

      sent_queue = []
      sentences = instance['refexp']
      for refexp in sentences:
        sent_queue += [refexp]

      err, all_moves, move_ids = get_moves(
          instance, xlongitude, ylatitude, sentences)

      if use_gt_moves and (err or all_moves == [] or len(all_moves[0]) != len(sentences)):
        count_err += 1
        continue

      pano = instance['img_idx']
      datum['gt_moves'] = all_moves
      datum['refexps'] = sentences
      all_sentences += sent_queue

      start_loc = instance['actions'][0]['act_deg_list'][0][0]
      start_x, start_y = get_coordinates(start_loc[0], start_loc[1],
                                         full_w=full_w,
                                         full_h=full_h)
      start_fov, _ = get_nearest(cached_nodes, start_x, start_y)
      gt_path = [start_fov]
      path = []
      intermediate_paths = []

      # add gt loc of waldo
      instance['actions'][0]['act_deg_list'][-1].append(
          [datum["gt_lng"], datum["gt_lat"]])

      if add_cached_path:
        for kk, act_list in enumerate(instance['actions'][0]['act_deg_list']):
          act = act_list[-1]
          lng, lat = act
          x, y = get_coordinates(lng, lat,
                                 full_w=full_w,
                                 full_h=full_h)

          min_n, _ = get_nearest(cached_nodes, x, y)
          gt_path.append(min_n)

        path = [gt_path[0]]
        for kk in range(len(gt_path)-1):
          start = gt_path[kk]
          end = gt_path[kk+1]
          intermediate_path = cached_paths[start][end]
          path += intermediate_path[1:]
          intermediate_paths.append(intermediate_path)
        assert(len(gt_path) <= len(
            path), 'len(gt_path) <= len(path) {} > {}'.format(len(gt_path), len(path)))
        assert(len(datum['refexps']) != len(intermediate_paths),
               'len(refepxs) != len(intermediate_paths)')
        datum['gt_path'] = gt_path
        datum['path'] = path
        datum['intermediate_paths'] = intermediate_paths
        datum['actionid'] = instance['actions'][0]['actionid']
      if task == 'continuous_grounding':
        data.append(datum)
      if task == 'graph_grounding' and task_root != '':
        node_path = os.path.join(task_root, '{}.npy'.format(pano))
        node_img = os.path.join(task_root, '{}.jpg'.format(pano))
        fov_prefix = os.path.join(task_root, '{}.fov'.format(pano))
        nodes = np.load(node_path, allow_pickle=True)[()]
        fovs, graph_hops, new_nodes = get_graph_hops(nodes,
                                                     instance['actions'],
                                                     datum['img_src'])
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
                                                     datum['img_src'])

        for moves, move_id in zip(all_moves, move_ids):
          for mm, move in enumerate(moves):
            mdatum = {}
            mdatum['move_id'] = mm
            mdatum['move_max'] = len(sentences)
            mdatum['img_src'] = datum['img_src']
            mdatum['actionid'] = move_id
            mdatum['annotationid'] = instance['annotationid']

            xlng, ylat = move[0], move[1]
            mx = int(full_w * ((xlng + 180)/360.0))
            my = int(full_h - full_h *
                     ((ylat + 90)/180.0))

            mdatum['xlongitude'] = move[0]
            mdatum['ylatitude'] = move[1]

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
      elif task == 'grid_fov_pretraining' and task_root != '':
        grid_nodes, _ = generate_grid(degree=degree)
        node_path = os.path.join(graph_root, '{}.npy'.format(pano))
        node_img = os.path.join(graph_root, '{}.jpg'.format(pano))
        nodes = np.load(node_path, allow_pickle=True)[()]

        for n in grid_nodes:
          node = grid_nodes[n]
          mdatum = {}
          fov_id = node['id']
          mdatum['fov_id'] = fov_id

          mdatum['move_max'] = len(sentences)
          mdatum['img_src'] = datum['img_src']
          # mdatum['actionid'] = move_id
          mdatum['annotationid'] = instance['annotationid']

          ylat, xlng = node['lat'], node['lng']
          mx, my = node['x'], node['y']
          mdatum['ylatitude'] = ylat
          mdatum['xlongitude'] = xlng
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
      elif task == 'balanced_fov_pretraining' and task_root != '':
        node_path = os.path.join(graph_root, '{}.npy'.format(pano))
        node_img = os.path.join(graph_root, '{}.jpg'.format(pano))
        nodes = np.load(node_path, allow_pickle=True)[()]

        node_path = os.path.join(task_root, '{}.npy'.format(pano))
        balanced_nodes = np.load(node_path, allow_pickle=True)[()]

        for n in balanced_nodes:
          node = balanced_nodes[n]

          mdatum = {}
          fov_id = node['id']
          mdatum['fov_id'] = fov_id

          mdatum['move_max'] = len(sentences)
          mdatum['img_src'] = datum['img_src']
          # mdatum['actionid'] = move_id
          mdatum['annotationid'] = instance['annotationid']

          ylat, xlng = node['lat'], node['lng']
          mx, my = node['x'], node['y']
          mdatum['xlongitude'] = xlng
          mdatum['ylatitude'] = ylat
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
      elif task == 'cached_fov_pretraining':
        node_path = os.path.join(graph_root, 'pano_{}.npy'.format(pano))
        node_img = os.path.join(graph_root, 'pano_{}.jpg'.format(pano))
        nodes = np.load(node_path, allow_pickle=True)[()]

        for n in cached_nodes:
          node = cached_nodes[n]
          mdatum = {}
          fov_id = node['idx']
          mdatum['fov_id'] = fov_id

          mdatum['move_max'] = len(sentences)
          mdatum['img_src'] = datum['img_src']
          # mdatum['actionid'] = move_id
          mdatum['annotationid'] = instance['annotationid']

          ylat, xlng = node['lat'], node['lng']
          mx, my = node['x'], node['y']
          mdatum['xlongitude'] = xlng
          mdatum['ylatitude'] = ylat
          mdatum['x'] = mx
          mdatum['y'] = my

          mdatum['refexps'] = sentences
          fov_file = os.path.join(cache_root, 'fovs',
                                  task_root, 'pano_{}.{}.jpg'.format(pano, fov_id))

          mdatum['fov_file'] = fov_file
          regions, obj_list = get_objects(mx, my, nodes, vg2idx,
                                          include_vectors=False)
          mdatum['regions'] = regions
          mdatum['obj_list'] = obj_list

          directions = [len(obj_list['navigation'][d])
                        for d in obj_list['navigation'].keys()]

          if sum(directions) > 0:
            data.append(mdatum)
    data_list.append(data)
    print('{} instances have errors'.format(count_err))
    pbar.close()

  print('_'*20)
  for c in stats:
    print('{:<16} {:2.2f}'.format(c, stats[c]))
  print('_'*20)

  n_instances = sum([len(l) for l in data_list])
  print('Dumping {} instances to {}'.format(n_instances, output_file))

  np.save(open(output_file, 'wb'), {'data_list': data_list,
                                    'sentences': all_sentences})
#  return data_list, all_sentences


def dump_mp3d_datasets(splits, output_file,
                       task='grid_fov_pretraining',
                       task_root='',
                       graph_root='',
                       obj_dict_file='../data/vg_object_dictionaries.top100.matterport3d.json',
                       image_list_file='../data/imagelist.matterport3d.txt',
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

  data_list = []
  data = []
  stats = defaultdict(int)
  image_list = [line.strip().split('.')[0]
                for line in open(image_list_file)]
  pbar = tqdm(image_list)

  for ii, pano in enumerate(pbar):

    if task == 'continuous_grounding':
      raise NotImplementedError()
    elif task == 'graph_grounding' and task_root != '':
      raise NotImplementedError()
    elif task == 'fov_pretraining' and task_root != '':
      raise NotImplementedError()
    elif task == 'grid_fov_pretraining' and task_root != '':
      grid_nodes, _ = generate_grid(degree=degree)
      node_path = os.path.join(graph_root, '{}.npy'.format(pano))
      nodes = np.load(node_path, allow_pickle=True)[()]

      for n in grid_nodes:
        node = grid_nodes[n]

        mdatum = {}
        fov_id = node['id']
        mdatum['fov_id'] = fov_id

        mdatum['img_src'] = pano

        ylat, xlng = node['lat'], node['lng']
        mx, my = node['x'], node['y']
        mdatum['ylatitude'] = ylat
        mdatum['xlongitude'] = xlng
        mdatum['x'] = mx
        mdatum['y'] = my

        mdatum['refexps'] = []
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
    elif task == 'balanced_fov_pretraining' and task_root != '':
      raise NotImplementedError()
    data_list.append(data)
  pbar.close()

  print('_'*20)
  for c in stats:
    print('{:<16} {:2.2f}'.format(c, stats[c]))
  print('_'*20)

  print('Dumping to {}'.format(output_file))
  np.save(open(output_file, 'wb'), {'data_list': data_list,
                                    'sentences': []})


def visualize_path(path, node_dict, edges, canvas, size=20, path_color=color_green):

  n_start = node_dict[path[0]]

  ox, oy = n_start['x'], n_start['y']
  add_square(canvas, ox, oy,
             size=size,
             color=color_white)
  prev_node = path[0]
  for n in path[1:]:

    node = node_dict[n]
    ox, oy = node['x'], node['y']
    add_square(canvas, ox, oy,
               size=size,
               color=color_yellow)

    if (prev_node, n) in edges:
      ns, ne = edges[prev_node, n]

      sx = ns[0]
      sy = ns[1]
      ex = ne[0]
      ey = ne[1]

      if 4552-72 <= sx <= 4552 <= ex:
        if sx < 4552:
          cv2.line(canvas,
                   (sx, sy),
                   (4552, ey),
                   path_color, 8, 8)
        cv2.line(canvas,
                 (0, sy),
                 (ex % 4552, ey),
                 path_color, 8, 8)
      elif sx <= 72 and 4552 <= ex:
        if sx > 0:
          cv2.line(canvas,
                   (sx, sy),
                   (0, ey),
                   path_color, 8, 8)
        cv2.line(canvas,
                 (4552, sy),
                 (4552 - ex % 4552, ey),
                 path_color, 8, 8)
      elif 4552 < sx:
        cv2.line(canvas,
                 (sx - 4552, sy),
                 (0, ey),
                 path_color, 8, 8)
      else:
        cv2.line(canvas,
                 (sx, sy),
                 (ex, ey),
                 path_color, 8, 8)
    prev_node = n

  n_end = node_dict[path[-1]]

  ox, oy = n_end['x'], n_end['y']
  add_square(canvas, ox, oy,
             size=size,
             color=color_red)
  return canvas


def add_square(canvas, x, y,
               size=20,
               color=(255, 255, 255)):
  canvas[y-size:y+size, x-size:x+size, :] = color


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


if __name__ == '__main__':
  dump_datasets(['train', 'dev', 'test'], 'all', 'dummy')
