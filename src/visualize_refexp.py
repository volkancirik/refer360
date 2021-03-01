from model_utils import get_det2_features
import os
import numpy as np
from utils import add_overlay
from utils import get_coordinates
from utils import get_nearest
from utils import visualize_path
from utils import add_square
from utils import color_red, color_green, color_blue
import json
import panoramic_camera_cached as camera
import random
import sys
import cv2
import matplotlib

usage = '''

argv[1] : annotation id, leave empty for a random one

examples:
PYTHONPATH=.. python visualize_refexp.py 7340
PYTHONPATH=.. python visualize_refexp.py
'''
matplotlib.use('Agg')


def load_datum(datum, cam):

  refer = ".".join([" ".join(refexp) for refexp in datum['refexp']])
  n_steps = len(datum['act_deg_list'][0])
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

  add_square(panels['original'], gt_x, gt_y, color=color_green)
  add_square(panels['original'], pred_x, pred_y, color=color_red)
  add_square(panels['grid'], gt_x, gt_y, color=color_green)
  add_square(panels['grid'], pred_x, pred_y, color=color_red)

  gt_path = []
  for kk, act in enumerate(datum['act_deg_list'][0]):
    lng, lat = act
    x, y = get_coordinates(lng, lat,
                           full_w=full_w,
                           full_h=full_h)

    min_n, _ = get_nearest(nodes, x, y)
    if len(gt_path) == 0 or gt_path[-1] != min_n:
      gt_path.append(min_n)
    if len(gt_path) >= 2:
      start = gt_path[-2]
      end = gt_path[-1]
      intermediate_path = cam.paths[start][end]
      panels['grid'] = visualize_path(
          intermediate_path, cam.nodes, cam.edges, panels['grid'])
  return image_path, gt_path, panels, waldo_position_lng, waldo_position_lat, pred_lng, pred_lat


if __name__ == '__main__':

  if len(sys.argv) > 2:
    print(usage)
    quit(1)

  full_w, full_h = 4552, 2276
  size = 20
  font = cv2.FONT_HERSHEY_SIMPLEX
  cache_root = '../data/cached_data_15degrees/'
  grid_path = cache_root + 'canvas.jpg'
  meta_file = os.path.join(cache_root, 'meta.npy')
  meta = np.load(meta_file, allow_pickle=True)[()]
  nodes = meta['nodes']
  waldo_img = cv2.resize(cv2.imread(
      '../data/waldo.png', cv2.IMREAD_COLOR), (60, 40), interpolation=cv2.INTER_AREA)
  target_img = cv2.resize(cv2.imread(
      '../data/target.png', cv2.IMREAD_COLOR), (60, 60), interpolation=cv2.INTER_AREA)

  data = json.load(open('../data/train.json'))
  act_instances = {}
  for instance in data:
    for action in instance['actions']:
      action['img_src'] = instance['img_src']
      action['refexp'] = instance['refexp']
      action['gt_longitude'] = instance['xlng_deg']
      action['gt_latitude'] = instance['ylat_deg']
      act_instances[action['actionid']] = action

  if len(sys.argv) == 2:
    actionid = int(sys.argv[1])
  else:
    actionid = random.choice(list(act_instances.keys()))
    print('randomly picked', actionid)

  cam = camera.CachedPanoramicCamera(cache_root)
  cam.load_maps()

  load_new = True
  while True:

    if load_new:
      load_new = False
      fov_id = 0
      datum = act_instances[actionid]
      image_path, gt_path, panels, waldo_position_lng, waldo_position_lat, pred_lng, pred_lat = load_datum(
          datum, cam)

      cam.load_img(image_path, convert_color=False)
      cam.load_fovs()
      n_steps = len(gt_path)
    cam.look_fov(gt_path[fov_id])
    img = cam.get_image()
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
    cv2.imshow("Use awsd to move c to close or quit", combined)

    key = cv2.waitKey()

    if key == 110:
      print('randomly picked', actionid)
      actionid = random.choice(list(act_instances.keys()))
      load_new = True
    elif key == 2:
      if fov_id > 0:
        fov_id = fov_id - 1
    elif key == 3:
      if fov_id < n_steps-1:
        fov_id = (fov_id + 1)
    if key == 99:  # c
      break