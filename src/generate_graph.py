import paths

from collections import defaultdict
import json
from operator import itemgetter

import cv2
import sys
import torch

from panoramic_camera_gpu import PanoramicCameraGPU as camera
#from panoramic_camera import PanoramicCamera as camera

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from tqdm import tqdm
import numpy as np
import os
from scipy.spatial import Delaunay

from utils import rad2degree
from utils import get_det2_features
from utils import generate_fovs
from utils import generate_grid

torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def get_triangulation(nodes, left_w, width):

  n_nodes = len(nodes)
  order2nid = {i: i for i in range(n_nodes)}

  idx = n_nodes
  new_nodes = nodes
  for ii, n in enumerate(nodes):
    if n[0] < left_w:
      order2nid[idx] = ii
      try:
        new_nodes.append((n[0]+width, n[1]))
      except:
        print('something is wrong with new_nodes')
        print(type(new_nodes))
        print(new_nodes)
        print(len(new_nodes))
        quit(1)
      idx += 1
  tri = Delaunay(np.array(new_nodes))
  return tri, order2nid, n_nodes


def generate_graph(image_path, predictor, vg_classes,
                   full_w=4552,
                   full_h=2276,
                   left_w=128,
                   topk=60):
  font = cv2.FONT_HERSHEY_SIMPLEX
  THRESHOLD = 50
  size = 20

  all_objects = []
  all_nodes = []
  all_scores = []

  slat, slng = rad2degree(np.random.uniform(
      0, 6), np.random.uniform(1, 1.5), adjust=True)
  sx = int(full_w * ((slat + 180)/360.0))
  sy = int(full_h - full_h *
           ((slng + 90)/180.0))

  all_objects.append((slat, slng, -1, sx, sy, [sx-1, sy-1, sx+1, sy+1]))
  all_nodes.append([sx, sy])
  all_scores = [1]

  for fov_size in [400, 1200]:

    cam = camera(output_image_shape=(fov_size, fov_size))
    cam.load_img(image_path)

    for lng in range(-45, 90, 45):
      for lat in range(-180, 180, 30):
        cam.look(lat, lng)
        pixel_map = cam.get_map()
        img = cam.get_image()
        detectron_outputs = predictor(img)

        boxes, object_types, det_scores = get_det2_features(
            detectron_outputs["instances"].to("cpu"))
        for b, o, s in zip(boxes, object_types, det_scores):
          center_x = int((b[0]+b[2])/2)
          center_y = int((b[1]+b[3])/2)
          o_lat = pixel_map[center_y][center_x][0]
          o_lng = pixel_map[center_y][center_x][1]

          flag = True
          for r in all_objects:
            d = ((r[0] - o_lng)**2 + (r[1] - o_lat)**2)**0.5
            if d < THRESHOLD and r[2] == o:
              flag = False
          if flag:
            x = int(full_w * ((o_lat + 180)/360.0))
            y = int(full_h - full_h *
                    ((o_lng + 90)/180.0))
            all_objects.append((o_lat, o_lng, o, x, y, b))
            all_nodes.append([x, y])
            all_scores.append(s)

  scores, nodes, objects = zip(
      *sorted(zip(all_scores, all_nodes, all_objects), reverse=True))

  scores = list(scores[:topk])
  nodes = list(nodes[:topk])
  objects = list(objects[:topk])

  print('# of detected objects {} | {} used objects'.format(
      len(all_nodes), len(nodes)))
  tri, order2nid, n_nodes = get_triangulation(nodes, left_w, full_w)
  canvas = np.zeros((full_h, full_w, 3), dtype='uint8')

  clr = (255, 0, 0)

  node_dict = dict()
  for kk, o in enumerate(objects):
    o_type, ox, oy = o[2], o[3], o[4]

    canvas[oy-size:oy+size, ox-size:ox+size, 0] = 255.
    canvas[oy-size:oy+size, ox-size:ox+size, 1:] = 0

    o_label = '<START>'
    if o_type > 0:
      o_label = vg_classes[o_type]

    cv2.putText(canvas, o_label, (ox+size, oy+size), font, 3, clr, 5)
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

    n0 = order2nid[s[0]]
    n1 = order2nid[s[1]]
    n2 = order2nid[s[2]]
    node_dict[n0]['neighbors'] += [n1]
    node_dict[n1]['neighbors'] += [n0]

    node_dict[n0]['neighbors'] += [n2]
    node_dict[n2]['neighbors'] += [n0]

    node_dict[n1]['neighbors'] += [n2]
    node_dict[n2]['neighbors'] += [n1]

    cv2.line(canvas, (x0, nodes[s[0]][1]),
             (x1, nodes[s[1]][1]), color1, 3, 8)
    cv2.line(canvas, (x0, nodes[s[0]][1]),
             (x2, nodes[s[2]][1]), color2, 3, 8)
    cv2.line(canvas, (x2, nodes[s[2]][1]),
             (x1, nodes[s[1]][1]), color3, 3, 8)

  canvas = cv2.resize(canvas, (800, 400))

  return node_dict, canvas


def run_generate_graph():

  data_path = '../py_bottom_up_attention/demo/data/genome/1600-400-20'
  vg_classes = []
  with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
      vg_classes.append(object.split(',')[0].lower().strip())

  MetadataCatalog.get("vg").thing_classes = vg_classes
  yaml_file = '../py_bottom_up_attention/configs/VG-Detection/faster_rcnn_R_101_C4_caffe.yaml'
  cfg = get_cfg()
  cfg.merge_from_file(yaml_file)
  cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 100
  cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

  cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe.pkl"
  predictor = DefaultPredictor(cfg)

  image_list = [line.strip()
                for line in open(sys.argv[1])]  # ../data/imagelist.txt
  image_root = sys.argv[2]  # ../data/refer360images
  out_root = sys.argv[3]  # ../data/graph_data

  if not os.path.exists(out_root):
    try:
      os.makedirs(out_root)
    except:
      print('Cannot create folder {}'.format(out_root))
      quit(1)

  pbar = tqdm(image_list)
  for fname in pbar:
    image_path = os.path.join(image_root, fname)
    pano = fname.split('/')[-1].split('.')[0]
    nodes, canvas = generate_graph(
        image_path, predictor, vg_classes)

    node_path = os.path.join(
        out_root, '{}.npy'.format(pano))

    np.save(open(node_path, 'wb'), nodes)
    cv2.imwrite(os.path.join(out_root, '{}.jpg'.format(pano)), canvas)
    fov_prefix = os.path.join(
        out_root, '{}.fov'.format(pano))

    generate_fovs(image_path, node_path, fov_prefix)


def run_generate_grid():

  image_list = [line.strip()
                for line in open(sys.argv[1])]  # ../data/imagelist.txt
  image_root = sys.argv[2]  # ../data/refer360images
  out_root = sys.argv[3]  # ../data/grid_data_30
  degree = int(sys.argv[4])  # 30

  if not os.path.exists(out_root):
    try:
      os.makedirs(out_root)
    except:
      print('Cannot create folder {}'.format(out_root))
      quit(1)

  pbar = tqdm(image_list)
  for fname in pbar:
    image_path = os.path.join(image_root, fname)
    pano = fname.split('/')[-1].split('.')[0]
    nodes, canvas = generate_grid(degree=degree)

    node_path = os.path.join(
        out_root, '{}.npy'.format(pano))

    np.save(open(node_path, 'wb'), nodes)

    fov_prefix = os.path.join(
        out_root, '{}.fov'.format(pano))

    generate_fovs(image_path, node_path, fov_prefix)


def generate_object_dictionaries():
  usage = '''Generates json file with dictionaries of visualgenome object ids to refer360 object ids conversions.
  PYTHONPATH=.. python generate_graph.py ../data/imagelist.txt ../data/graph_data/  ../data/vg_object_dictionaries.all.json all

  PYTHONPATH=.. python generate_graph.py ../data/imagelist.txt ../data/graph_data/  ../data/vg_object_dictionaries.[table,chair].json table,chair
'''
  if len(sys.argv) != 5:
    print(usage)
    quit(1)
  data_path = '../py_bottom_up_attention/demo/data/genome/1600-400-20'
  vg_classes = []
  with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
      vg_classes.append(object.split(',')[0].lower().strip())

  image_list = [line.strip()
                for line in open(sys.argv[1])]  # ../data/imagelist.txt
  out_root = sys.argv[2]  # ../data/graph_data

  if not os.path.exists(out_root):
    try:
      os.makedirs(out_root)
    except:
      print('Cannot create folder {}'.format(out_root))
      quit(1)

  counts = defaultdict(int)
  pbar = tqdm(image_list)
  for fname in pbar:
    pano = fname.split('/')[-1].split('.')[0]
    node_path = os.path.join(
        out_root, '{}.npy'.format(pano))

    nodes = np.load(node_path, allow_pickle=True)[()]
    for node in nodes:
      vg_id = nodes[node]['obj_id']
      counts[vg_id] += 1

  sorted_counts = sorted(counts.items(), key=itemgetter(1))[::-1]

  min_th = 5
  max_th = 50000

  vg2idx = {}
  idx2vg = {}

  idx = 0
  out_file = sys.argv[3]  # '../data/vg_object_dictionaries.all.json'
  include = set(sys.argv[4].split(','))  # all or table,chair

  for ii, (vg_idx, cnt) in enumerate(sorted_counts):
    if min_th <= cnt <= max_th and (vg_classes[vg_idx] in include or sys.argv[4] == 'all'):
      print(idx, cnt, vg_classes[vg_idx])
      vg2idx[vg_idx] = idx
      idx2vg[idx] = vg_idx
      idx += 1
  json.dump({'vg2idx': vg2idx,
             'idx2vg': idx2vg}, open(out_file, 'w'))


if __name__ == '__main__':
  run_generate_graph()
  # run_generate_grid()
  # generate_object_dictionaries()
