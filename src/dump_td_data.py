import paths
import argparse
from pprint import pprint
from utils_td import dump_td_datasets, SPLITS
import os

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=['continuous_grounding',
                                       'cached_fov_pretraining'],
                    default='continuous_grounding',
                    help='task name, default: continuous_grounding')
parser.add_argument('--dump_root', type=str,
                    required=True, help='Dump folder path')
parser.add_argument('--task_root', type=str,  default='',
                    help='FoVs path for <task>, default=""')
parser.add_argument('--graph_root', type=str,  default='',
                    help='object detections folder for panoramas, default=""')
parser.add_argument('--obj_dict_file',
                    type=str,
                    default='../data/vg_object_dictionaries.all.json',
                    help='object dictionaries, default=../data/vg_object_dictionaries.all.json')
parser.add_argument('--cache_root',
                    type=str,
                    default='../data/td_cached_data_30degrees/',
                    help='cache root, default=../data/td_cached_data_30degrees/')
parser.add_argument('--degree', type=int,  default=15,
                    help='degrees between fovs, default="15')

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
      args.dump_root, '{}.[all].imdb.npy'.format(split_name))
  dump_td_datasets([split_name], dump_name,
                   task=args.task,
                   task_root=args.task_root,
                   graph_root=args.graph_root,
                   obj_dict_file=args.obj_dict_file,
                   cache_root=args.cache_root,
                   degree=args.degree)
