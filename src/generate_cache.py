from utils import get_coordinates
from utils import add_square
from utils import color_gray
import networkx as nx
from panoramic_camera import PanoramicCamera as camera
from torchvision import transforms
import torch
import torchvision.models as models
from tqdm import tqdm
import os
import cv2
import sys
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
plt.rcdefaults()


sys.path.append('../src')

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def generate_maps(image_path, nodes, map_prefix,
                  full_w=4552,
                  full_h=2276,
                  fov_size=400):

  cam = camera(output_image_shape=(fov_size, fov_size))
  cam.load_img(image_path)

  pbar = tqdm(nodes)
  for n in pbar:
    node = nodes[n]
    idx = node['idx']
    xlng = node['lng']
    ylat = node['lat']

    cam.look(xlng, ylat)
    lng_map, lat_map = cam.get_pixel_map()
    inverse_lng_map, inverse_lat_map = cam.get_inverse_map()

    dummy = np.zeros((full_h, full_w), dtype='uint8')
    m1 = np.greater(inverse_lng_map, dummy)
    m2 = np.greater(inverse_lat_map, dummy)
    m = np.logical_or(m1, m2).astype(np.uint8)*255
    mask = np.stack((m,)*3, axis=-1)

    mask_file = os.path.join(map_prefix, '{}.mask.jpg'.format(idx))
    cv2.imwrite(mask_file, mask)
    datum = {
        'lng_map': lng_map,
        'lat_map': lat_map,
        'inverse_lng_map': inverse_lng_map,
        'inverse_lat_map': inverse_lat_map
    }
    maps_file = os.path.join(map_prefix, '{}.map.npy'.format(idx))
    np.save(maps_file, datum)


def dump_fovs(cam, image_path, map_path, nodes, fov_prefix,
              fov_size=400):

  cam.load_img(image_path)

  for jj, n in enumerate(nodes.keys()):
    idx = nodes[n]['idx']
    cam.look_fov(idx)
    fov_img = cam.get_image()
    fov_file = fov_prefix + '.{}.jpg'.format(idx)
    pil_img = Image.fromarray(fov_img)
    pil_img.save(fov_file)


def dump_features(model, preprocess, nodes, features_prefix, fov_prefix):

  feats = []
  for jj, n in enumerate(sorted(nodes.keys())):
    idx = nodes[n]['idx']

    fov_file = fov_prefix + '.{}.jpg'.format(idx)
    image_obj = Image.open(fov_file)
    img_tensor = preprocess(image_obj)

    feat_fov = model(img_tensor.to(DEVICE).unsqueeze(
        0)).cpu().detach().numpy()
    feats.append(feat_fov)
  features_file = features_prefix + '.npy'
  all_fovs_feats = np.concatenate(feats, axis=0)
  np.save(features_file, all_fovs_feats)


def generate_grid(full_w=4552,
                  full_h=2276,
                  degree=5,
                  size=10):
  left_w = int(full_w * (degree/360)+1)

  dx = full_w * (degree/360)
  dy = full_h * (degree/180)
  DISTANCE = (dx ** 2 + dy ** 2) ** 0.5 + 10

  node_dict = dict()
  objects = []
  nodes = []
  positions = {}
  kk = 0

  for lat in range(75, -75, -degree):
    for lng in range(-180, 180, degree):
      x, y = get_coordinates(lng, lat,
                             full_w=full_w,
                             full_h=full_h)
      objects.append((lng, lat, 1, x, y, []))
      nodes.append([x, y])
      positions[kk] = [x, y]
      n = {
          'idx': kk,
          'lng': lng,
          'lat': lat,
          'obj_label': '',  # grid fov label
          'obj_id': '1',  # grid fov
          'x': x,
          'y': y,
          'boxes': [],
          'neighbors': [],
          'dir2neighbor': {},
          'neighbor2dir': {}
      }
      node_dict[kk] = n
      kk += 1

  canvas = np.zeros((full_h, full_w, 3), dtype='uint8')

  n_nodes = len(nodes)
  order2nid = {i: i for i in range(n_nodes)}

  idx = n_nodes
  new_nodes = nodes
  for ii, n in enumerate(nodes):
    if n[0] < left_w:
      order2nid[idx] = ii
      new_nodes.append((n[0]+full_w, n[1]))
      idx += 1

  edges = {}
  G = nx.Graph()

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

        if s1[0] < s2[0] and s1[1] < s2[1]:
          direction = 'dr'
          inverse_dir = 'ul'
        elif s1[0] < s2[0] and s1[1] == s2[1]:
          direction = 'r'
          inverse_dir = 'l'
        elif s1[0] < s2[0] and s1[1] > s2[1]:
          direction = 'ur'
          inverse_dir = 'dl'
        elif s1[0] == s2[0] and s1[1] < s2[1]:
          direction = 'd'
          inverse_dir = 'r'
        elif s1[0] == s2[0] and s1[1] > s2[1]:
          direction = 'u'
          inverse_dir = 'd'
        elif s1[0] > s2[0] and s1[1] < s2[1]:
          direction = 'dl'
          inverse_dir = 'ur'
        elif s1[0] > s2[0] and s1[1] == s2[1]:
          direction = 'l'
          inverse_dir = 'r'
        elif s1[0] > s2[0] and s1[1] > s2[1]:
          direction = 'ul'
          inverse_dir = 'dr'

        node_dict[n0]['dir2neighbor'][direction] = n1
        node_dict[n1]['dir2neighbor'][inverse_dir] = n0

        node_dict[n0]['neighbor2dir'][n1] = direction
        node_dict[n1]['neighbor2dir'][n0] = inverse_dir

        edges[n0, n1] = ((s1[0], s1[1]), (s2[0], s2[1]))
        cv2.line(canvas, (s1[0], s1[1]),
                 (s2[0], s2[1]), color_gray, 3, 8)
        G.add_edge(n0, n1, weight=d)
  for kk, o in enumerate(objects):
    ox, oy = o[3], o[4]

    add_square(canvas, ox, oy, size=10, color=color_gray)
  nx.set_node_attributes(G, values=positions, name='position')
  paths = dict(nx.all_pairs_dijkstra_path(G))
  distances = dict(nx.all_pairs_dijkstra_path_length(G))
  return node_dict, canvas, edges, G, paths, distances


def run_generate_cache():

  image_list = [line.strip()
                for line in open(sys.argv[1])]  # ../data/imagelist.txt
  image_root = sys.argv[2]  # ../data/refer360images
  out_root = sys.argv[3]  # ../data/cached_data_15degrees
  degree = int(sys.argv[4])  # 15
  full_w = int(sys.argv[5])  # 4552
  full_h = int(sys.argv[6])  # 2276
  fov_size = int(sys.argv[7])  # 400

  if not os.path.exists(out_root):
    try:
      os.makedirs(out_root)
      os.makedirs(os.path.join(out_root, 'maps'))
      os.makedirs(os.path.join(out_root, 'fovs'))
      os.makedirs(os.path.join(out_root, 'features'))
    except:
      print('Cannot create folder {}'.format(out_root))
      quit(1)
  print('Generating grid')
  nodes, canvas, edges, graph, paths, distances = generate_grid(degree=degree,
                                                                full_h=full_h,
                                                                full_w=full_w)

  meta = {
      'nodes': nodes,
      'edges': edges,
      'graph': graph,
      'paths': paths,
      'distances': distances
  }

  map_prefix = os.path.join(
      out_root, 'maps')
  meta_file = os.path.join(out_root, 'meta.npy')
  np.save(meta_file, meta)
  cv2.imwrite(os.path.join(out_root, 'canvas.jpg'), canvas)
  image_path = os.path.join(image_root, image_list[0])
  # print('Generating maps')
  # generate_maps(image_path, nodes, map_prefix,
  #               fov_size=fov_size,
  #               full_h=full_h,
  #               full_w=full_w)

  cam = camera(output_image_shape=(fov_size, fov_size))
  cam.load_maps(out_root)
  print('Generating fovs')
  pbar = tqdm(image_list)
  for fname in pbar:
    image_path = os.path.join(image_root, fname)
    pano = fname.split('/')[-1].split('.')[0]
    fov_prefix = os.path.join(
        out_root, 'fovs', '{}'.format(pano))
    dump_fovs(cam, image_path, out_root, nodes, fov_prefix,
              fov_size=fov_size)

  # resnetmodel = models.resnet152(pretrained=True).to(DEVICE)
  # model = torch.nn.Sequential(
  #     *(list(resnetmodel.children())[:-1])).to(DEVICE)
  # for p in model.parameters():
  #   model.requires_grad = False
  # preprocess = transforms.Compose([
  #     transforms.ToTensor(),
  #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
  #                          std=[0.229, 0.224, 0.225]),
  # ])

  # print('Generating features')
  # pbar = tqdm(image_list)
  # for fname in pbar:
  #   pano = fname.split('/')[-1].split('.')[0]
  #   features_prefix = os.path.join(
  #       out_root, 'features', '{}'.format(pano))
  #   fov_prefix = os.path.join(
  #       out_root, 'fovs', '{}'.format(pano))
  #   dump_features(model, preprocess, nodes, features_prefix, fov_prefix)


if __name__ == '__main__':
  '''Example run:
       python generate_cache.py ../data/imagelist.txt ../data/refer360images ../data/cached_data_15degrees 15 4552 2276 400
  '''
  run_generate_cache()
