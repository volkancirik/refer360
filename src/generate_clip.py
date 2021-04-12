# from utils import get_coordinates
# from utils import add_square
# from utils import color_gray
# import networkx as nx
# from panoramic_camera import PanoramicCamera as camera
# import cv2

import clip
import torch

from tqdm import tqdm
import os

import sys
import numpy as np
from PIL import Image


sys.path.append('../src')

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def dump_features(model, preprocess, nodes, features_prefix, fov_prefix,
                  device='cpu'):

  feats = []
  for jj, n in enumerate(sorted(nodes.keys())):
    idx = nodes[n]['idx']

    fov_file = fov_prefix + '.{}.jpg'.format(idx)
    image = preprocess(Image.open(fov_file)).unsqueeze(0).to(device)
    feat_fov = model.encode_image(image).cpu().detach().numpy()
    feats.append(feat_fov)
  features_file = features_prefix + '.clip.npy'
  all_fovs_feats = np.concatenate(feats, axis=0)

  np.save(features_file, all_fovs_feats)


def run_generate_clip():

  out_root = sys.argv[1]  # ../data/cached_data_15degrees
  image_list = [line.strip()
                for line in open(sys.argv[2])]  # ../data/imagelist.txt

  meta_file = os.path.join(out_root, 'meta.npy')
  meta = np.load(meta_file, allow_pickle=True)[()]
  nodes = meta['nodes']

  device = "cuda" if torch.cuda.is_available() else "cpu"
  model, preprocess = clip.load("ViT-B/32", device=device)

  print('Generating clip features')
  pbar = tqdm(image_list)
  for fname in pbar:
    pano = fname.split('/')[-1].split('.')[0]
    features_prefix = os.path.join(
        out_root, 'features', '{}'.format(pano))
    fov_prefix = os.path.join(
        out_root, 'fovs', '{}'.format(pano))
    dump_features(model, preprocess, nodes, features_prefix, fov_prefix,
                  device=device)


if __name__ == '__main__':
  '''Example run:
       python generate_clip.py ../data/cached_data_15degrees ../data/imagelist.txt 
  '''
  run_generate_clip()
