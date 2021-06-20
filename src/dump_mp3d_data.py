import paths
import sys
from utils_mp3d import dump_mp3d_datasets

import os

dump_root = sys.argv[1]
task = sys.argv[2]
task_root = sys.argv[3]
graph_root = sys.argv[4]
obj_dict_file = sys.argv[5]

print('task', task)
print('task_root', task_root)
print('graph_root', graph_root)
print('obj_dict_file', obj_dict_file)

split_name = 'dummy'
images = 'all'

if dump_root != '' and not os.path.exists(dump_root):
  try:
    os.makedirs(dump_root)
  except:
    print('Cannot create folder {}'.format(dump_root))
    quit(1)


dump_mp3d_datasets(dump_root,
                   task=task,
                   task_root=task_root,
                   graph_root=graph_root,
                   obj_dict_file=obj_dict_file)
