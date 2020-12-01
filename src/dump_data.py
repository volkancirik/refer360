usage = '''Dump data for a category of images. all for all categories.

argv[1] : data dump folder
argv[2] : list of image categories comma separated or 'all'
argv[3] : graph data root (optional)
example:
$ PYTHONPATH=.. python dump_data.py  ../data/dumps all
$ PYTHONPATH=.. python dump_data.py  ../data/graph_grounding all graph_grounding ../data/graph_data
$ PYTHONPATH=.. python dump_data.py  ../data/fov_pretraining all fov_pretraining ../data/fov_data ../data/graph_data ../data/vg_object_dictionaries.all.json
$ PYTHONPATH=.. python dump_data.py  ../data/dumps restaurant
$ PYTHONPATH=.. python dump_data.py  ../data/dumps restaurant,shop,expo_showroom,living_room,bedroom
$ PYTHONPATH=.. python dump_data.py  ../data/dumps street,plaza_courtyard
'''
from utils import dump_datasets
from utils import SPLITS
import sys
import os
if len(sys.argv) < 2 or len(sys.argv) > 7:
  print(usage)
  quit(1)
task = 'continuous_grounding'
task_root = ''
graph_root = ''
obj_dict_file = '../data/vg_object_dictionaries.all.json'

if len(sys.argv) > 4:
  task = sys.argv[3]
if len(sys.argv) == 5:
  task_root = sys.argv[4]
if 6 <= len(sys.argv) <= 7:
  task_root = sys.argv[4]
  if 'fov_pretraining' in task:
    graph_root = sys.argv[5]
    if len(sys.argv) == 7:
      obj_dict_file = sys.argv[6]
  else:
    print(usage)
    quit(1)


dump_root = sys.argv[1]
images = sys.argv[2]

if dump_root != '' and not os.path.exists(dump_root):
  try:
    os.makedirs(dump_root)
  except:
    print('Cannot create folder {}'.format(dump_root))
    quit(1)

print('images', images)
print('task', task)
print('task_root', task_root)
print('graph_root', graph_root)
print('obj_dict_file', obj_dict_file)

for ii, split_name in enumerate(SPLITS):
  dump_name = os.path.join(
      dump_root, '{}.[{}].imdb.npy'.format(split_name, images))
  dump_datasets([split_name], images, dump_name,
                task=task,
                task_root=task_root,
                graph_root=graph_root,
                obj_dict_file=obj_dict_file)
