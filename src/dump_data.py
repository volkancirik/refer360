import paths
import sys
import argparse
from pprint import pprint
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
parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=['continuous_grounding',
                                       'graph_grounding',
                                       'fov_pretraining',
                                       'grid_fov_pretraining'],
                    default='continuous_grounding',
                    help='task name, default: continuous_grounding')
parser.add_argument('--dump_root', type=str, required=True, help='Dump folder path')
parser.add_argument("--images",
                    type=str,
                    default='all',
                    help='list of image categories comma separated or all')
parser.add_argument('--task_root', type=str,  default='', help='FoVs path for <task>, default=""')
parser.add_argument('--graph_root', type=str,  default='', help='object detections folder for panoramas, default=""')
parser.add_argument('--obj_dict_file',
                    type=str,
                    default='../data/vg_object_dictionaries.all.json',
                    help='object dictionaries, default=../data/vg_object_dictionaries.all.json')
parser.add_argument('--cache_root',
                    type=str,
                    default='../data/cached_data_15degrees/',
                    help='cache root, default=../data/cached_data_15degrees/')
args = parser.parse_args()

if args.dump_root != '' and not os.path.exists(args.dump_root):
  try:
    os.makedirs(args.dump_root)
  except:
    print('Cannot create folder {}'.format(args.dump_root))
    quit(1)

pprint(args)
for ii, split_name in enumerate(SPLITS):
  dump_name = os.path.join(
      args.dump_root, '{}.[{}].imdb.npy'.format(split_name, args.images))
  dump_datasets([split_name], args.images, dump_name,
                task=args.task,
                task_root=args.task_root,
                graph_root=args.graph_root,
                obj_dict_file=args.obj_dict_file,
                cache_root=args.cache_root)
