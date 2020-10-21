usage = '''Dump data for a category of images. all for all categories.

argv[1] : data dump folder
argv[2] : list of image categories comma separated or 'all'
argv[3] : graph data root (optional)
example:
$ PYTHONPATH=.. python dump_data.py  ../data/dumps all
$ PYTHONPATH=.. python dump_data.py  ../data/dumps all ../data/graph_data
$ PYTHONPATH=.. python dump_data.py  ../data/dumps restaurant
$ PYTHONPATH=.. python dump_data.py  ../data/dumps restaurant,shop,expo_showroom,living_room,bedroom
$ PYTHONPATH=.. python dump_data.py  ../data/dumps street,plaza_courtyard
'''
from utils import dump_datasets

import sys
import os
splits = [
    'validation.seen',
    'validation.unseen',
    'test.seen',
    'test.unseen',
    'train',
]
if len(sys.argv) < 2 or len(sys.argv) > 4:
  print(usage)
  quit(1)
graph_root = ''
if len(sys.argv) == 4:
  graph_root = sys.argv[3]
dump_root = sys.argv[1]
images = sys.argv[2]
for ii, split_name in enumerate(splits):
  dump_name = os.path.join(
      dump_root, '{}.[{}].imdb.npy'.format(split_name, images))
  dump_datasets([split_name], images, dump_name,
                graph_root=graph_root)
