import paths

from collections import defaultdict
import json
from operator import itemgetter

import sys

from tqdm import tqdm
import numpy as np
import os

usage = '''Generates json file with dictionaries of visualgename object ids to refer360 object ids conversions.
       python generate_object_dictionaries.py ../data/imagelist.txt ../data/graph_data/  ../data/vg_object_dictionaries.all.json all
python generate_object_dictionaries.py ../data/imagelist.txt ../data/graph_data/  ../data/vg_object_dictionaries.[table,chair].json table,chair
       python generate_object_dictionaries.py ../data/imagelist.txt ../data/graph_data/  ../data/vg_object_dictionaries.top100.json top100
'''


def generate_object_dictionaries():
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

  topk = len(sorted_counts)
  include_all = True
  include = set()
  if 'top' in sys.argv[4]:
    topk = int(sys.argv[4].split('top')[1])
    print('will dump top{}'.format(topk))
  elif 'all' == sys.argv[4]:
    pass
  else:
    include_all = False
    include = set(sys.argv[4].split(','))  # all or table,chair

  for ii, (vg_idx, cnt) in enumerate(sorted_counts[:topk]):
    if min_th <= cnt <= max_th and (vg_classes[vg_idx] in include or include_all):
      print(idx, cnt, vg_classes[vg_idx])
      vg2idx[vg_idx] = idx
      idx2vg[idx] = vg_idx
      idx += 1

  obj_classes = ['']*len(idx2vg)
  for idx in idx2vg:
    obj_classes[int(idx)] = vg_classes[idx2vg[idx]]

  json.dump({'vg2idx': vg2idx,
             'idx2vg': idx2vg,
             'obj_classes': obj_classes
             }, open(out_file, 'w'))


if __name__ == '__main__':
  generate_object_dictionaries()
