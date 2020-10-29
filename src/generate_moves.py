usage = '''Generate ground-truth FoV moves.
argv[1] : data dump folder
'''
import json
import os
import sys
from tqdm import tqdm
from utils import generate_gt_moves
from utils import SPLITS, CAT2LOC
from utils import rad2degree, get_moves


def generate_all_moves(task_root,
                       splits=SPLITS,
                       image_categories='all'):
  if task_root != '' and not os.path.exists(task_root):
    try:
      os.makedirs(task_root)
    except:
      print('Cannot create folder {}'.format(task_root))
      quit(1)

  banned_turkers = set(['onboarding', 'vcirik'])

  image_set = set(CAT2LOC.keys())
  if image_categories != 'all':
    image_set = set(image_categories.split(','))
    for image in image_set:
      if image not in CAT2LOC:
        raise NotImplementedError(
            'Image Category {} is not in the dataset'.format(image))

  for split in splits:

    instances = json.load(open('../data/{}.json'.format(split), 'r'))

    pbar = tqdm(instances)
    for ii, instance in enumerate(pbar):
      if instance['turkerid'] not in banned_turkers:
        if 'ann_cat' in instance and instance['ann_cat'] == 'M' and split == 'train':
          continue

        latitude, longitude = rad2degree(instance['latitude'],
                                         instance['longitude'],
                                         adjust=True)

        img_category = '_'.join(instance['imageurl'].split(
            '/')[-1].split('.')[0].split('_')[2:])
        pano_name = "_".join(instance['imageurl'].split(
            '/')[-1].split('.')[0].split('_')[:2]) + ".jpg"

        if img_category not in CAT2LOC or img_category not in image_set:
          continue
        img_loc = CAT2LOC[img_category]
        image_path = '../data/refer360images/{}/{}/{}'.format(
            img_loc, img_category, pano_name)

        sentences = []
        for refexp in instance['refexp'].replace('\n', '').split('|||')[1:]:
          if refexp[-1] not in ['.', '!', '?', '\'']:
            refexp += '.'
          sentences.append(refexp.split())

        err, all_moves, move_ids = get_moves(
            instance, latitude, longitude, len(sentences))

        if err or all_moves == [] or len(all_moves[0]) != len(sentences):
          continue

        generate_gt_moves(image_path, all_moves, move_ids, fov_root=task_root)


if __name__ == '__main__':
  if len(sys.argv) != 2:
    print(usage)
    quit(1)

  generate_all_moves(sys.argv[1])
