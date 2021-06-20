import cv2
import json
import numpy as np
import os
from tqdm import tqdm

# from panoramic_camera import PanoramicCamera as camera
from collections import defaultdict

from utils import generate_grid
from utils import get_objects
R2R_SPLIT_NAMES = ['train',
                   'val_seen',
                   'val_unseen',
                   'test']


def get_mp3d_dictionaries(dict_file='../data/R2R_scan2split_dicts.npy'):
  '''Loads dictionaries for viewpoint and scan data
  scan      -> split,
  split     -> scan
  scan      -> viewpoint
  viewpoint -> scan
  split     -> viewpoint
  viewpoint -> split
  '''
  dicts = np.load(dict_file, allow_pickle=True)[()]

  return dicts


def dump_mp3d_datasets(output_root,
                       task='grid_fov_pretraining',
                       task_root='',
                       graph_root='',
                       obj_dict_file='../data/vg_object_dictionaries.top100.matterport3d.json',
                       image_list_file='../data/imagelist.matterport3d.txt',
                       degree=45):
  '''Prepares and dumps dataset to an npy file.
  '''
  if task_root != '' and not os.path.exists(task_root):
    try:
      os.makedirs(task_root)
    except:
      print('Cannot create folder {}'.format(task_root))
      quit(1)

  dicts = get_mp3d_dictionaries()
  vg2idx = json.load(open(obj_dict_file, 'r'))['vg2idx']

  data_list = defaultdict(list)
  data = []

  image_list = [line.strip().split('.')[0]
                for line in open(image_list_file)]
  pbar = tqdm(image_list)

  for ii, pano in enumerate(pbar):
    splits = dicts['viewpoint2split'][pano]
    if task == 'grid_fov_pretraining' and task_root != '':
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
          for split in splits:
            data_list[split].append(mdatum)
    else:
      raise NotImplementedError()

  pbar.close()

  for split in R2R_SPLIT_NAMES:
    output_file = os.path.join(
        output_root, '{}.[all].imdb.npy'.format(split))

    print('Dumping {} instances to {}'.format(
        len(data_list[split]), output_file))
    np.save(open(output_file, 'wb'), {'data_list': [data_list[split]],
                                      'sentences': []})
