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
from utils_td import SPLITS
from utils import CAT2LOC, load_datasets
import json
import panoramic_camera_cached as camera
import random
import sys
import cv2
import matplotlib
from pprint import pprint
usage = '''

argv[1] : path to output folder
argv[2] : image category separated by ',' restaurant,shop,expo_showroom,living_room,bedroom,street,plaza_courtyard
argv[3] : (optional) annotation id, leave empty for a random one

examples:
 python visualize_refexp.py visualize_out all 7340
 python visualize_refexp.py visualize_out restaurant,living_room
'''
matplotlib.use('Agg')


def randomly_pick_datum(act_instances, categories):
  while True:
    actionid = random.choice(list(act_instances.keys()))
    return actionid


def load_datum(datum, cam):

  refer = "\n".join([" ".join(refexp) for refexp in datum['refexp']])
  # pprint(datum)
  n_steps = len(datum['act_deg_list'])
  image_path = datum['img_src']

  waldo_position_lng = datum['gt_longitude']
  waldo_position_lat = datum['gt_latitude']

  pred_lng = datum['gt_longitude']
  pred_lat = datum['gt_latitude']

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

  #add_square(panels['original'], gt_x, gt_y, color=color_green)
  #add_square(panels['original'], pred_x, pred_y, color=color_red)
  #add_square(panels['grid'], gt_x, gt_y, color=color_green)
  #add_square(panels['grid'], pred_x, pred_y, color=color_red)

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

  if len(sys.argv) > 4 or len(sys.argv) <= 2:
    print(usage)
    quit(1)

  full_w, full_h = 3000, 1500
  size = 20
  font = cv2.FONT_HERSHEY_SIMPLEX
  cache_root = '../data/cached_td_data30degrees'
  grid_path = os.path.join(cache_root, 'canvas.jpg')
  meta_file = os.path.join(cache_root, 'meta.npy')
  meta = np.load(meta_file, allow_pickle=True)[()]
  nodes = meta['nodes']
  waldo_img = cv2.resize(cv2.imread(
      '../data/waldo.png', cv2.IMREAD_COLOR), (60, 40), interpolation=cv2.INTER_AREA)
  target_img = cv2.resize(cv2.imread(
      '../data/target.png', cv2.IMREAD_COLOR), (60, 60), interpolation=cv2.INTER_AREA)

  act_instances = {}
  pano2act = defaultdict(list)
  pano2ann = defaultdict(list)

  data_list, _ = load_datasets(SPLITS,
                               root='../data/td_continuous_grounding')
  for data in data_list:
    for instance in data:
      for action in instance['actions']:
        action['img_src'] = instance['img_src']
        action['refexp'] = instance['refexps']
        action['gt_longitude'] = instance['gt_lng']
        action['gt_latitude'] = instance['gt_lat']
        image_path = instance['img_src']
        pano_id = image_path.split('/')[-1]

        act_instances[action['actionid']] = action

        pano2act[pano_id].append(action['actionid'])
        pano2ann[pano_id].append(action['actionid'])
  dump_root = sys.argv[1]
  if dump_root != '' and not os.path.exists(dump_root):
    try:
      os.makedirs(dump_root)
    except:
      print('Cannot create folder {}'.format(dump_root))

  if sys.argv[2] == 'all':
    categories = set(CAT2LOC.keys())
  else:
    categories = set()
    for cat in sys.argv[2].split(','):
      if cat not in CAT2LOC:
        print('{} not a part of image categories'.format(cat))
        quit(1)
      categories.add(cat)
  print('image categories:', categories)
  if len(sys.argv) == 4:
    actionid = int(sys.argv[3])
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

    cam.look_fov(gt_path[fov_id])
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

    for jj, refexp in enumerate(datum['refexp']):
      refer = " ".join(refexp)
      cv2.putText(img, refer, (5, (jj+2)*10), font, 0.35, (0, 0, 255), 1)

    grid = panels['grid'].copy()
    add_square(grid, curr_x, curr_y, color=color_blue)
    mask = cam.get_mask()
    grid = cv2.bitwise_and(grid, mask)

    original = cv2.resize(panels['original'], (800, 400))
    grid = cv2.resize(grid, (800, 400))
    mask = cv2.resize(panels['mask'], (800, 400))
    semantic = cv2.resize(panels['semantic'], (800, 400))
    resized_img = cv2.resize(img, (800, 800))

    column1 = np.concatenate((original, grid), axis=0)
    column2 = np.concatenate((mask, semantic), axis=0)
    combined = np.concatenate((resized_img, column1, column2), axis=1)
    cv2.imshow(
        "Use left/right arrows to move FoV. (n)ext (p)revious example. (c)lose", combined)
    prefix = '{}.{}'.format(actionid, fov_id)
    for (d, f) in [(original, 'original'),
                   (grid, 'grid'),
                   (mask, 'mask'),
                   (semantic, 'semantic')]:
      fname = os.path.join(dump_root, prefix + '.' + f + '.png')
      cv2.imwrite(fname, d)

    image_path = datum['img_src']
    pano_id = image_path.split('/')[-1]
    print('pano_id:', pano_id)
    print('actions:', pano2act[pano_id])

    key = cv2.waitKey()
    if key == 110:
      demo_id = demo_id + 1
      if demo_id == len(demo_list):
        print('randomly picked', actionid)
        #actionid = random.choice(list(act_instances.keys()))
        actionid = randomly_pick_datum(act_instances, categories)
        demo_list.append(actionid)
      else:
        actionid = demo_list[demo_id]
      load_new = True
    elif key == 112:
      demo_id = max(demo_id - 1, 0)
      actionid = demo_list[demo_id]
      load_new = True
    elif key == 2:
      if fov_id > 0:
        fov_id = fov_id - 1
    elif key == 3:
      if fov_id < n_steps-1:
        fov_id = (fov_id + 1)
    if key == 99:  # c
      break
# actions: [7851, 7805, 8160, 9952, 10309, 10659, 13339, 18016, 17797, 19009, 20677, 21314]
