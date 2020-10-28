import cv2
import sys

import panoramic_camera as camera
import json

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from tqdm import tqdm
import numpy as np
import os
from scipy.spatial import Delaunay

cat2loc = {'restaurant': 'indoor',
           'shop': 'indoor',
           'expo_showroom': 'indoor',
           'living_room': 'indoor',
           'bedroom': 'indoor',
           'street': 'outdoor',
           'plaza_courtyard': 'outdoor',
           }


def get_triangulation(nodes, left_w, width):

  n_nodes = len(nodes)
  order2nid = {i: i for i in range(n_nodes)}

  idx = n_nodes
  new_nodes = nodes
  for ii, n in enumerate(nodes):
    if n[0] < left_w:
      order2nid[idx] = ii
      new_nodes.append((n[0]+width, n[1]))
      idx += 1
  tri = Delaunay(np.array(new_nodes))
  return tri, order2nid, n_nodes


def get_det2_features(detections):
  boxes = []
  obj_classes = []

  for box, obj_class in zip(detections.pred_boxes, detections.pred_classes):
    box = box.cpu().detach().numpy()
    boxes.append(box)
    obj_class = obj_class.cpu().detach().numpy().tolist()
    obj_classes.append(obj_class)
  return boxes, obj_classes


def rad2degree(lat, lng):

  lng = np.degrees(lng + 0.1)
  lat = np.degrees(lat - 0.015)
  if lat > 180:
    lat = lat - 360
  return lat, lng


def run_history(history, predictor, vg_classes,
                full_w=4552,
                full_h=2276,
                fov_size=512,
                left_w=128):
  font = cv2.FONT_HERSHEY_SIMPLEX
  THRESHOLD = 20
  size = 20

  img_category = '_'.join(history['imageurl'].split(
      '/')[-1].split('.')[0].split('_')[2:])
  pano_name = "_".join(history['imageurl'].split(
      '/')[-1].split('.')[0].split('_')[:2]) + ".jpg"

  img_loc = cat2loc[img_category]
  image_path = '../data/sun360_originals/{}/{}/{}'.format(
      img_loc, img_category, pano_name)

  fovs = []

  if len(history['actions']) < 1:
    return [], [], []

  objects = []
  nodes = []

  slat, slng = rad2degree(np.random.uniform(0, 6), np.random.uniform(1, 1.5))
  sx = int(full_w * ((slat + 180)/360.0))
  sy = int(full_h - full_h *
           ((slng + 90)/180.0))

  objects.append((slat, slng, 'START', sx, sy))
  nodes.append([sx, sy])

  print(history['actions'][0].keys())
  print('_'*10)
  for ii, inst in enumerate(history['actions'][0]['action_list'].split('_')):
    for jj, fov in enumerate(inst.split('|')[1:]):
      rlat, rlng = float(fov.split(',')[0]), float(fov.split(',')[1])
      lat, lng = rad2degree(rlat, rlng)
      fx = int(full_w * ((lat + 180)/360.0))
      fy = int(full_h - full_h *
               ((lng + 90)/180.0))

      fovs.append((lat, lng, '{}.{}'.format(ii, jj), fx, fy))

  waldo_position_lat, waldo_position_lng = rad2degree(
      history['longitude'], history['latitude'])
  wx = int(full_w * ((waldo_position_lat + 180)/360.0))
  wy = int(full_h - full_h *
           ((waldo_position_lng + 90)/180.0))

  cam = camera.PanoramicCamera(output_image_shape=(fov_size, fov_size))
  cam.load_img(image_path, lng=0,
               lat=0)

  for lng in range(-45, 90, 45):
    for lat in range(-180, 180, 45):
      cam.look(lat, lng)
      pixel_map = cam.get_map()
      img = cam.get_image()

      detectron_outputs = predictor(img)

      boxes, object_types = get_det2_features(
          detectron_outputs["instances"].to("cpu"))
      for b, o in zip(boxes, object_types):
        center_x = int((b[0]+b[2])/2)
        center_y = int((b[1]+b[3])/2)
        o_lat = pixel_map[center_y][center_x][0]
        o_lng = pixel_map[center_y][center_x][1]

        flag = True
        for r in objects:
          d = ((r[0] - o_lng)**2 + (r[1] - o_lat)**2)**0.5
          if d < THRESHOLD and r[2] == o:
            flag = False
        if flag:
          x = int(full_w * ((o_lat + 180)/360.0))
          y = int(full_h - full_h *
                  ((o_lng + 90)/180.0))
          objects.append((o_lat, o_lng, o, x, y, b))

          nodes.append([x, y])

  # fov to closest object
  closest_distances = []
  closest_nodes = []
  for fov in fovs:
    closest_d = full_h + full_w

    _, _, _, fx, fy = fov[0], fov[1], fov[2], fov[3], fov[4]

    for jj, obj in enumerate(objects):
      ox, oy = obj[3], obj[4]

      d = ((fx-ox)**2 + (fy-oy)**2)**0.5
      if d < closest_d:
        closest_d = d
        closest = jj
    closest_distances.append(closest_d)
    closest_nodes.append(closest)

  closest_waldo_d = full_w + full_h
  for jj, obj in enumerate(objects):
    ox, oy = obj[3], obj[4]
    d = ((wx-ox)**2 + (wy-oy)**2)**0.5
    if d < closest_waldo_d:
      closest_waldo_d = d

  fov_distances = [full_w*full_h]*len(fovs)

  for kk, fov in enumerate(fovs):

    o = objects[closest_nodes[kk]]
    lat, lng, _ = o[0], o[1], o[2]
    cam.look(lat, lng)

    fov_coor = cam.get_image_coordinate_for(
        fov[0], fov[1])
    if fov_coor is not None:
      x = fov_coor[0]
      y = fov_coor[1]

      cx = int(fov_size / 2)
      cy = int(fov_size / 2)
      d = ((x-cx)**2 + (y-cy)**2)**0.5

      if fov_distances[kk] > d:
        fov_distances[kk] = d

  tri, order2nid, n_nodes = get_triangulation(nodes, left_w, full_w)
  canvas = np.zeros((full_h, full_w, 3), dtype='uint8')

  clr = (255, 0, 0)

  for kk, o in enumerate(objects):
    o_type, ox, oy = o[2], o[3], o[4]

    canvas[oy-size:oy+size, ox-size:ox+size, 0] = 255.
    canvas[oy-size:oy+size, ox-size:ox+size, 1:] = 0

    if type(o_type) == int:
      o_type = vg_classes[o_type]

    cv2.putText(canvas, o_type, (ox+size, oy+size), font, 3, clr, 5)

  clr = (0, 165, 255)
  for kk, fov in enumerate(fovs):
    fname, fx, fy = fov[2], fov[3], fov[4]

    canvas[fy-size:fy+size, fx-size:fx+size, 0] = 0
    canvas[fy-size:fy+size, fx-size:fx+size, 1] = 165
    canvas[fy-size:fy+size, fx-size:fx+size, 2] = 255

    cv2.putText(canvas, fname, (fx+size, fy+size), font, 3, clr, 5)

  canvas[wy-size:wy+size, wx-size:wx+size, 2] = 255.
  canvas[wy-size:wy+size, wx-size:wx+size, :2] = 0

  debug_color = (0, 0, 256)
  color = (125, 125, 125)
  for s in tri.simplices:

    x0 = nodes[s[0]][0]
    color1 = color2 = color3 = color
    if x0 > full_w:
      x0 -= full_w
      color1 = debug_color
      color2 = debug_color

    x1 = nodes[s[1]][0]
    if x1 > full_w:
      x1 -= full_w
      color1 = debug_color
      color3 = debug_color

    x2 = nodes[s[2]][0]
    if x2 > full_w:
      x2 -= full_w
      color2 = debug_color
      color3 = debug_color

    cv2.line(canvas, (x0, nodes[s[0]][1]),
             (x1, nodes[s[1]][1]), color1, 3, 8)
    cv2.line(canvas, (x0, nodes[s[0]][1]),
             (x2, nodes[s[2]][1]), color2, 3, 8)
    cv2.line(canvas, (x2, nodes[s[2]][1]),
             (x1, nodes[s[1]][1]), color3, 3, 8)

  canvas = cv2.resize(canvas, (800, 400))
  cv2.imshow("", canvas)
  key = cv2.waitKey()

  return [closest_waldo_d], closest_distances, fov_distances


if __name__ == '__main__':

  data_path = '/projects2/touchdown/py_bottom_up_attention/demo/data/genome/1600-400-20'
  vg_classes = []
  with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
      vg_classes.append(object.split(',')[0].lower().strip())

  MetadataCatalog.get("vg").thing_classes = vg_classes
  yaml_file = '/projects2/touchdown/py_bottom_up_attention/configs/VG-Detection/faster_rcnn_R_101_C4_caffe.yaml'
  cfg = get_cfg()
  cfg.merge_from_file(yaml_file)
  cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
  cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6

  cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe.pkl"
  predictor = DefaultPredictor(cfg)

  closest_waldo = []
  closest_fov = []
  closest_full = []

  d = json.load(open(sys.argv[1], 'r'))[::-1][10:20]

  pbar = tqdm(d)
  for history in pbar:
    wd, fulld, fovd = run_history(history, predictor, vg_classes)
    closest_waldo += wd
    closest_full += fulld
    closest_fov += fovd

  json.dump({'w': closest_waldo,
             'fovs': closest_fov,
             'full': closest_full
             }, open('analysis/{}.json'.format(sys.argv[2]), 'w'))
