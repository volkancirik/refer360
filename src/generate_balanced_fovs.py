import paths

from PIL import Image
import cv2
import sys
import torch
from sklearn.cluster import KMeans
from panoramic_camera_gpu import PanoramicCameraGPU as camera
# from panoramic_camera import PanoramicCamera as camera

from tqdm import tqdm
import numpy as np
import os
import random
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def generate_balanced_fovs(image_path, nodes, fov_prefix,
                           full_w=4552,
                           full_h=2276,
                           fov_size=400,
                           n_fovs=50):
  cam = camera(output_image_shape=(fov_size, fov_size))
  cam.load_img(image_path)

  size = 20
  canvas = np.zeros((full_h, full_w, 3), dtype='uint8')
  THRESHOLD = 100
  fov_nodes = []

  oclr = (255, 0, 0)
  fclr = (0, 0, 255)
  font = cv2.FONT_HERSHEY_SIMPLEX
  for jj, n in enumerate(nodes.keys()):
    idx, olat, olng, ox, oy = nodes[n]['id'], nodes[n]['lat'], nodes[n]['lng'], nodes[n]['x'], nodes[n]['y']
    if idx == 0:
      continue

    cam.look(olat, olng)
    canvas[oy-size:oy+size, ox-size:ox+size, 0] = 255.
    canvas[oy-size:oy+size, ox-size:ox+size, 1:] = 0

    cv2.putText(canvas, str(idx), (ox+size, oy+size), font, 3, oclr, 5)

    for x_coeff in np.arange(-1.5, 2, 1):
      for y_coeff in np.arange(-1.5, 2, 1):
        if not x_coeff and not y_coeff:
          continue

        x_diff = fov_size*x_coeff + random.randint(0, size)
        y_diff = fov_size*y_coeff + random.randint(0, size)

        if ox+x_diff > full_w:
          new_x = (ox + x_diff) - full_w
        elif ox+x_diff < 0:
          new_x = 2*full_w - (ox + x_diff)
        else:
          new_x = ox + x_diff

        if oy+y_diff > full_h:
          new_y = (oy + y_diff) - full_h
        elif oy+y_diff < 0:
          new_y = 2*full_h - (oy + y_diff)
        else:
          new_y = oy + y_diff

        new_x = int(new_x)
        new_y = int(new_y)
        # flag = True
        # for fov in fov_nodes:
        #   px, py = fov[0], fov[1]
        #   d = ((px - new_x)**2 + (py - new_y)**2)**0.5
        #   if d < THRESHOLD:
        #     flag = False
        #     break
        # if flag:
        fov_nodes.append([new_x, new_y])
        canvas[new_y-size:new_y+size, new_x-size:new_x+size, :2] = 0
        canvas[new_y-size:new_y+size, new_x-size:new_x+size, 2] = 255
        # cv2.putText(canvas, 'x{}|y{}'.format(x_coeff, y_coeff),
        #             (new_x+size, new_y+size), font, 3, fclr, 5)
  fov_data = np.array(fov_nodes)

  kmeans = KMeans(n_clusters=n_fovs, random_state=0).fit(fov_data)

  fov_nodes = kmeans.cluster_centers_.tolist()
  node_dict = {}
  for jj, fov in enumerate(fov_nodes):
    x, y = int(fov[0]), int(fov[1])
    canvas[y-size:y+size, x-size:x+size, 0] = 0
    canvas[y-size:y+size, x-size:x+size, 1] = 255
    canvas[y-size:y+size, x-size:x+size, 2] = 0

    lat = (x*1.0/full_w)*360 - 180
    lng = (full_h - y*1.0)*(180./full_h) - 90

    cam.look(lat, lng)
    #pixel_map = cam.get_map()
    # test_lat = pixel_map[200][200][0].item()
    # test_lng = pixel_map[200][200][1].item()

    fov_img = cam.get_image()
    fov_file = fov_prefix + '{}.jpg'.format(jj)
    pil_img = Image.fromarray(fov_img)
    pil_img.save(fov_file)

    n = {
        'id': jj,
        'lat': lat,
        'lng': lng,
        'obj_label': 'fov',
        'obj_id': 0,
        'x': x,
        'y': y,
        'boxes': [],
        'neighbors': []
    }
    node_dict[jj] = n

  canvas = cv2.resize(canvas, (800, 400))
  return node_dict, canvas


def run_generate_balanced_fovs():
  image_list = [line.strip()
                for line in open(sys.argv[1])]  # ../data/imagelist.txt
  image_root = sys.argv[2]  # ../data/refer360images
  graph_root = sys.argv[3]  # ../data/graph_data
  out_root = sys.argv[4]  # ../data/balanced_fov_pretraining_data
  n_fovs = int(sys.argv[5])  # 50

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

    org_node_path = os.path.join(
        graph_root, '{}.npy'.format(pano))
    org_nodes = np.load(org_node_path,
                        allow_pickle=True)[()]

    fov_prefix = os.path.join(
        out_root, '{}.fov'.format(pano))

    new_nodes, new_canvas = generate_balanced_fovs(
        image_path, org_nodes, fov_prefix, n_fovs=n_fovs)
    new_node_path = os.path.join(
        out_root, '{}.npy'.format(pano))
    np.save(open(new_node_path, 'wb'), new_nodes)
    new_canvas_file = os.path.join(
        out_root, '{}.jpg'.format(pano))
    cv2.imwrite(new_canvas_file, new_canvas)
    pbar.set_description('{:3.3f} | {:3.3f}'.format(
        len(new_nodes), len(org_nodes)))


if __name__ == '__main__':
  '''Example run:
       python generate_balanced_fovs.py ../data/imagelist.txt ../data/refer360images ../data/graph_data ../data/balanced_fov_data
  '''

  run_generate_balanced_fovs()
