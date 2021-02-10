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
#from panoramic_camera import PanoramicCamera as camera
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
    'canonical': ['up', 'down', 'left', 'right'],
    'cartesian': ['vertical', 'horizontal'],
    'lup': ['lateral', 'up', 'down'],
    'canonical_proximity': ['close_up', 'close_down', 'close_left', 'close_right',
                            'far_up', 'far_down', 'far_left', 'far_right']
}


def get_object_dictionaries(obj_dict_file):
  '''Loads object object dictionaries
  visual genome -> idx2
  idx ->  visual genome
  idx -> object classes'''

  data = json.load(
      open(obj_dict_file, 'r'))
  vg2idx = data['vg2idx']
  idx2vg = data['idx2vg']
  obj_classes = data['obj_classes']

  return vg2idx, idx2vg, obj_classes


def get_objects(move_x, move_y, nodes, vg2idx,
                full_w=4552,
                full_h=2276,
                fov_size=400):
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
      #cv2.imwrite(fov_file, fov_img)
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
    idx, lat, lng = nodes[n]['id'], nodes[n]['lat'], nodes[n]['lng']
    cam.look(lat, lng)

    fov_img = cam.get_image()
    fov_file = fov_prefix + '{}.jpg'.format(idx)
    #cv2.imwrite(fov_file, fov_img)
    pil_img = Image.fromarray(fov_img)
    pil_img.save(fov_file)


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
            print('\n>>>>>>')
            pprint(mdatum)
            quit(1)
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

        mdatum['pano'] = pano

        lat, lng = node['lat'], node['lng']
        mx, my = node['x'], node['y']
        mdatum['latitude'] = lat
        mdatum['longitude'] = lng
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
