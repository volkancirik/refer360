import paths

import sys
from tqdm import tqdm
import numpy as np
import os
from utils import generate_fovs
from utils import generate_grid
import torch

torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')


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


if __name__ == '__main__':
  '''Example run:
       python generate_grid.py ../data/imagelist.txt ../data/refer360images ../data/grid_data_30degrees 30
  '''
  run_generate_grid()
