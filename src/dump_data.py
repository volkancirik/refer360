usage = '''Dump data for a category of images. all for all categories.

argv[1] : data dump folder
argv[2] : list of image categories comma separated or 'all'
argv[3] : graph data root (optional)
example:
$ PYTHONPATH=.. python dump_data.py  ../data/dumps all
$ PYTHONPATH=.. python dump_data.py  ../data/dumps all graph_grounding ../data/graph_data
$ PYTHONPATH=.. python dump_data.py  ../data/dumps all fov_pretraining ../data/fov_data
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
if len(sys.argv) < 2 or len(sys.argv) > 5:
  print(usage)
  quit(1)
task = 'continuous_grounding'
task_root = ''
if len(sys.argv) > 4:
  task = sys.argv[3]
if len(sys.argv) == 5:
  task_root = sys.argv[4]

dump_root = sys.argv[1]
images = sys.argv[2]

if dump_root != '' and not os.path.exists(dump_root):
  try:
    os.makedirs(dump_root)
  except:
    print('Cannot create folder {}'.format(dump_root))
    quit(1)

print('task | task_root', task, task_root)
for ii, split_name in enumerate(splits):
  dump_name = os.path.join(
      dump_root, '{}.[{}].imdb.npy'.format(split_name, images))
  dump_datasets([split_name], images, dump_name,
                task=task,
                task_root=task_root)
