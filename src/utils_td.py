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
from utils import get_coordinates
from utils import get_nearest
from utils import coordinate2degrees
# list of dataset splits
SPLITS = [
    'dev',
    'test',
    'train',
]

color_gray = (125, 125, 125)
color_dark = (25, 25, 25)

color_red = (255, 0, 0)
color_green = (0, 255, 0)
color_blue = (0, 0, 255)
color_yellow = (255, 255, 0)
color_white = (255, 255, 255)


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


def dump_td_datasets(splits, output_file,
                     task='continuous_grounding',
                     task_root='',
                     graph_root='',
                     full_w=3000,
                     full_h=1500,
                     obj_dict_file='../data/vg_object_dictionaries.all.json',
                     degree=30,
                     cache_root='../data/cached_td_data30degrees/'):  # FIX
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
    raise NotImplementedError()

  data_list = []

  all_sentences = []

  stats = defaultdict(int)
  for split in splits:
    data = []

    td_data_file = '../data/td_data/{}.json'.format(split)
    lines = [(ii, line) for ii, line in enumerate(open(td_data_file))]
    pbar = tqdm(lines)
    count_err = 0
    for ii, line in pbar:
      datum = {}

      instance = {}
      td_instance = json.loads(line)

      instance['img_idx'] = td_instance['main_pano']
      instance['img_cat'] = 'street'
      instance['img_loc'] = 'outdoor'
      instance['img_src'] = '../data/td_data/images/' + \
          td_instance['main_pano'] + '.jpg'
      instance['annotationid'] = td_instance['route_id']
      center = json.loads(td_instance['main_static_center'])
      gt_coors = [int(center['y']*full_h), int(center['x']*full_w)]
      xlng_deg, ylat_deg = coordinate2degrees(gt_coors[1], gt_coors[0],
                                              full_w=full_w,
                                              full_h=full_h)
      instance['xlng_deg'], instance['ylat_deg'] = xlng_deg, ylat_deg
      instance['refexp'] = [td_instance['td_location_text'].split(' ')]

      n_rows = int(360 / degree)
      start_fov = np.random.randint(n_rows)
      start_node = cached_nodes[start_fov]
      assert start_node['idx'] == start_fov
      instance['actions'] = [{'act_deg_list': [[[start_node['lng'], start_node['lat']], [xlng_deg, ylat_deg]]],
                              'actionid': instance['annotationid']
                              }]
      all_moves = [
          [(start_node['lng'], start_node['lat']), (xlng_deg, ylat_deg)]]

      xlongitude, ylatitude = instance['xlng_deg'], instance['ylat_deg']

      datum['annotationid'] = instance['annotationid']
      datum["gt_lng"] = xlongitude
      datum["gt_lat"] = ylatitude

      img_cat = instance['img_cat']
      img_loc = instance['img_loc']
      img_src = instance['img_src']

      datum['img_src'] = img_src
      datum['img_category'] = img_cat
      stats[img_loc] += 1
      stats[img_cat] += 1

      sent_queue = []
      sentences = instance['refexp']
      for refexp in sentences:
        sent_queue += [refexp]

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
        assert len(gt_path) <= len(
            path), 'len(gt_path) <= len(path) {} > {}'.format(len(gt_path), len(path))
        assert len(datum['refexps']) == len(
            intermediate_paths), 'len(refepxs) != len(intermediate_paths) {} != {}'.format(len(datum['refexps']), len(intermediate_paths))
        datum['gt_path'] = gt_path
        datum['path'] = path
        datum['intermediate_paths'] = intermediate_paths
        datum['actionid'] = instance['actions'][0]['actionid']
        datum['actions'] = instance['actions']
      if task == 'continuous_grounding':
        data.append(datum)
    data_list.append(data)
    print('{} instances have errors'.format(count_err))
    pbar.close()

  n_instances = sum([len(l) for l in data_list])
  print('Dumping {} instances to {}'.format(n_instances, output_file))
  np.save(open(output_file, 'wb'), {'data_list': data_list,
                                    'sentences': all_sentences})


if __name__ == '__main__':
  dump_td_datasets(['train', 'dev', 'test'], 'dummy')
