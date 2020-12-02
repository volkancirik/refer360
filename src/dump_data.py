import paths
import sys
from utils import dump_datasets
from utils import SPLITS
import os


usage = '''Dump data for a category of images. all for all categories.

argv[1] : data dump folder
argv[2] : list of image categories comma separated or 'all'
argv[3] : task name continuous_grounding|graph_grounding|fov_pretraining|grid_fov_pretraining (optional)
argv[4] : task root in ../data where FoVs are located (optional)
argv[5] : object detections folder for panoramas ../data (optional)
argv[6] : object dictionary file
example:
$ python dump_data.py  ../data/continuous_grounding all
$ python dump_data.py  ../data/graph_grounding all graph_grounding ../data/graph_data_top60
$ python dump_data.py  ../data/fov_pretraining all fov_pretraining ../data/grount_truth_moves ../data/graph_data_top60 ../data/vg_object_dictionaries.top50.json
$ python dump_data.py  ../data/grid_fov_pretraining all grid_fov_pretraining ../data/grid_data_30degrees/ ../data/graph_data_top60 ../data/vg_object_dictionaries.top50.json
$ python dump_data.py  ../data/continuous_grounding restaurant
$ python dump_data.py  ../data/continuous_grounding restaurant,shop,expo_showroom,living_room,bedroom
$ python dump_data.py  ../data/continuous_grounding street,plaza_courtyard
'''

if len(sys.argv) < 2 or len(sys.argv) > 7:
  print(usage)
  quit(1)
task = 'continuous_grounding'
task_root = ''
graph_root = ''
obj_dict_file = '../data/vg_object_dictionaries.all.json'

dump_root = sys.argv[1]
images = sys.argv[2]

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
