from collections import defaultdict
from torch.utils.data import Dataset
from utils import DIRECTIONS, CAT2LOC
from utils import get_objects_classes, load_obj_tsv
from operator import itemgetter
import numpy as np
import os
import torch
import random
from tqdm import tqdm


def load_fovpretraining_splits(splits,
                               direction,
                               data_root='../data/fov_pretraining',
                               images='all',
                               task='task1',
                               obj_classes=[],
                               ignore_list=''):

  fov_files = []
  regions = []
  refexps = []
  obj_lists = []
  obj_queries = []
  obj_directions = []
  fovs = []

  labels = {}
  pano_metas = []

  ignore_list = ['move'+ignore
                 for ignore in ignore_list.split(',')] if ignore_list else []
  print('Following moves will be ignored: {}'.format(' , '.join(ignore_list)))

  for split in splits:
    split_file = os.path.join(
        data_root, '{}.[{}].imdb.npy'.format(split, images))
    print('loading split {} from {} for FoV pretraining'.format(split, split_file))
    instances = np.load(split_file, allow_pickle=True)[()]['data_list'][0]

    pbar = tqdm(instances)
    for instance in pbar:
      ignored = [ignore in instance['fov_file'] for ignore in ignore_list]

      if any(ignored):
        continue
      fov_files += [instance['fov_file']]
      pano_category = instance['pano'].split('/')[-2]
      pano_id = "_".join(instance['pano'].split(
          '/')[-1].split('.')[0].split('_')[:2])
      pano_loc = CAT2LOC[pano_category]

      pano_metas += [(pano_id, pano_category, pano_loc)]
      fovs += [(instance['latitude'], instance['longitude'])]
      try:
        refexps += [instance['refexp']]
      except:
        refexps += [instance['refexps']]
        pass

      regions += [instance['regions']]
      obj_lists += [instance['obj_list']]

      for dir_id, dir in enumerate(DIRECTIONS[direction]):
        for obj in instance['obj_list'][direction][dir]:
          obj_queries += [obj]
          obj_directions += [dir_id]

          if obj not in labels:
            labels[obj] = defaultdict(int)
          labels[obj][dir] += 1.0
    pbar.close()

  most_freq = -1
  if 'train' in splits:
    print('_'*20)
    for o, _ in enumerate(obj_classes):
      sorted_counts = sorted(labels[o].items(), key=itemgetter(1))[::-1]
      total = sum([cnt for _, cnt in sorted_counts])
      print('object:', obj_classes[o], 'total count:', int(total))
      for ii, (d, cnt) in enumerate(sorted_counts):
        print('{:20}:{:3.3f}'.format(d, cnt/total))
      print('_'*20)

      most_freq = DIRECTIONS[direction].index(sorted_counts[0][0])

  print('Loaded {} insances using {} direction method'.format(
      len(fov_files), direction))
  return fovs, fov_files, regions, refexps, obj_lists, obj_queries, obj_directions, labels, pano_metas, most_freq


class FoVPretrainingDataset(Dataset):
  def __init__(self, splits, direction,
               data_root='../data/fov_pretraining',
               images='all',
               task='task1',
               obj_dict_file='../data/vg_object_dictionaries.json',
               use_objects=False,
               ignore_list=''):

    vg2idx, idx2vg, obj_classes = get_objects_classes(obj_dict_file)
    self.vg2idx = vg2idx
    self.idx2vg = idx2vg
    self.obj_classes = obj_classes

    fovs, fov_files, regions, refexps, obj_lists, obj_queries, obj_directions, labels, pano_metas, most_freq = load_fovpretraining_splits(splits, direction,
                                                                                                                                          data_root=data_root,
                                                                                                                                          images=images,
                                                                                                                                          task=task,
                                                                                                                                          obj_classes=obj_classes,
                                                                                                                                          ignore_list=ignore_list)

    self.ignore_list = ignore_list
    self.direction = direction
    self.fovs = fovs
    self.fov_files = fov_files
    self.regions = regions
    self.refexps = refexps
    self.obj_lists = obj_lists
    self.obj_queries = obj_queries
    self.obj_directions = obj_directions
    self.task = task
    self.use_objects = use_objects
    self.labels = labels
    self.pano_metas = pano_metas
    self.most_freq = most_freq

    self.pano2idx = {p[0]: idx for idx, p in enumerate(pano_metas)}
    if self.use_objects:
      butd_feats = '/projects2/lxmert/data/refer360_fovpretraining_imgfeat/all_obj36.tsv'
      img_data = []
      img_data.extend(load_obj_tsv(butd_feats,
                                   topk=None))

      self.imgid2img = {}
      for img_datum in img_data:
        self.imgid2img[img_datum['img_id']] = img_datum

  def __getitem__(self, index):
    fov_file = self.fov_files[index]
    img_feat_path = fov_file.replace('.jpg', '.feat.npy')
    img_features = np.load(img_feat_path)
    image = torch.cuda.FloatTensor(img_features)
    return image

  def __len__(self):
    return len(self.fov_files)


class FoVTask1(FoVPretrainingDataset):
  def __init__(self, splits, direction,
               data_root='../data/fov_pretraining',
               images='all',
               obj_dict_file='../data/vg_object_dictionaries.json',
               use_objects=False,
               ignore_list=''):
    super(FoVTask1, self).__init__(splits, direction,
                                   data_root=data_root,
                                   images=images,
                                   task='task1',
                                   obj_dict_file=obj_dict_file,
                                   use_objects=use_objects,
                                   ignore_list=ignore_list)

  def __getitem__(self, index):

    (lat, lng) = self.fovs[index]
    latitude = torch.cuda.FloatTensor([lat])
    longitude = torch.cuda.FloatTensor([lng])

    fov_file = self.fov_files[index]
    regions = self.regions[index]

    img_feat_path = fov_file.replace('.jpg', '.feat.npy')
    img_features = np.load(img_feat_path)

    n_objects = len(self.vg2idx)

    image = torch.cuda.FloatTensor(img_features)

    targets = []
    for dir in DIRECTIONS[self.direction]:
      targets.append(torch.cuda.FloatTensor(
          regions[self.direction][dir]).resize(1, n_objects))
    target = torch.cat(targets,
                       axis=1).squeeze(0)

    if self.use_objects:
      img_id = fov_file.split('/')[-1].replace('.jpg', '')
      img_info = self.imgid2img[img_id]

      boxes = torch.cuda.FloatTensor(img_info['boxes'].copy())
      feats = torch.cuda.FloatTensor(img_info['features'].copy())

      img_h, img_w = img_info['img_h'], img_info['img_w']
      boxes[..., (0, 2)] /= img_w
      boxes[..., (1, 3)] /= img_h

      return latitude, longitude, image, boxes, feats, torch.tensor([]), '', target
    return latitude, longitude, image, torch.tensor([]), torch.tensor([]), torch.tensor([]), '', target


class FoVTask2(FoVPretrainingDataset):
  def __init__(self, splits, direction,
               data_root='../data/fov_pretraining',
               images='all',
               obj_dict_file='../data/vg_object_dictionaries.json',
               use_objects=False,
               ignore_list=''):
    super(FoVTask2, self).__init__(splits, direction,
                                   data_root=data_root,
                                   images=images,
                                   task='task2',
                                   obj_dict_file=obj_dict_file,
                                   use_objects=use_objects,
                                   ignore_list=ignore_list)

  def __getitem__(self, index):

    (lat, lng) = self.fovs[index]

    latitude = torch.cuda.FloatTensor([lat])
    longitude = torch.cuda.FloatTensor([lng])

    fov_file = self.fov_files[index]

    obj_query = self.obj_queries[index]
    obj_direction = self.obj_directions[index]

    img_feat_path = fov_file.replace('.jpg', '.feat.npy')
    img_features = np.load(img_feat_path)

    image = torch.cuda.FloatTensor(img_features)
    query = torch.cuda.LongTensor([obj_query])
    direction = torch.cuda.FloatTensor([obj_direction])

    if self.use_objects:
      img_id = fov_file.split('/')[-1].replace('.jpg', '')
      img_info = self.imgid2img[img_id]

      boxes = torch.cuda.FloatTensor(img_info['boxes'].copy())
      feats = torch.cuda.FloatTensor(img_info['features'].copy())

      img_h, img_w = img_info['img_h'], img_info['img_w']
      boxes[..., (0, 2)] /= img_w
      boxes[..., (1, 3)] /= img_h

      return latitude, longitude, image, boxes, feats, query, self.obj_classes[query], direction
    return latitude, longitude, image, torch.tensor([]), torch.tensor([]), query, self.obj_classes[query], direction


class FoVTask3(FoVPretrainingDataset):

  def __init__(self, splits, direction,
               data_root='../data/fov_pretraining',
               images='all',
               obj_dict_file='../data/vg_object_dictionaries.json',
               use_objects=False,
               obj_complement='pano',
               ignore_list=''):
    super(FoVTask3, self).__init__(splits, direction,
                                   data_root=data_root,
                                   images=images,
                                   task='task2',
                                   obj_dict_file=obj_dict_file,
                                   use_objects=use_objects,
                                   ignore_list=ignore_list)

    self.obj_complement = obj_complement
    self.pano2fov = defaultdict(set)
    self.cat2fov = defaultdict(set)
    self.loc2fov = defaultdict(set)

    self.obj2pano = defaultdict(set)
    self.obj2cat = defaultdict(set)
    self.obj2loc = defaultdict(set)

    self.fov2obj = defaultdict(set)

    for fov_file, pano_meta, obj_list in zip(self.fov_files, self.pano_metas, self.obj_lists):
      pano_id, pano_category, pano_loc = pano_meta

      self.pano2fov[pano_id].add(fov_file)
      self.cat2fov[pano_category].add(fov_file)
      self.loc2fov[pano_loc].add(fov_file)

      for dir in obj_list[self.direction]:
        for obj in obj_list[self.direction][dir]:

          self.obj2pano[obj].add(pano_id)
          self.obj2cat[obj].add(pano_category)
          self.obj2loc[obj].add(pano_loc)

          self.fov2obj[fov_file].add(obj)

  def __getitem__(self, index):

    (lat, lng) = self.fovs[index]
    latitude = torch.cuda.FloatTensor([lat])
    longitude = torch.cuda.FloatTensor([lng])

    fov_file = self.fov_files[index]

    obj_query = self.obj_queries[index]

    img_feat_path = fov_file.replace('.jpg', '.feat.npy')
    img_features = np.load(img_feat_path)

    image = torch.cuda.FloatTensor(img_features)
    query = torch.cuda.LongTensor([obj_query])

    pano_id, pano_category, pano_loc = self.pano_metas[index]

    diff_set = list(set([int(key) for key in self.idx2vg.keys()]
                        ).difference(self.fov2obj[fov_file]))
    if random.randint(0, 1) == 1 and len(diff_set) > 0:
      obj_query = random.choice(diff_set)
      query = torch.cuda.LongTensor([obj_query])
      presence = torch.cuda.FloatTensor([0])
    else:
      presence = torch.cuda.FloatTensor([1])

    if self.use_objects:
      img_id = fov_file.split('/')[-1].replace('.jpg', '')
      img_info = self.imgid2img[img_id]

      boxes = torch.cuda.FloatTensor(img_info['boxes'].copy())
      feats = torch.cuda.FloatTensor(img_info['features'].copy())

      img_h, img_w = img_info['img_h'], img_info['img_w']
      boxes[..., (0, 2)] /= img_w
      boxes[..., (1, 3)] /= img_h

      return latitude, longitude, image, boxes, feats, query, self.obj_classes[query], presence
    return latitude, longitude, image, torch.tensor([]), torch.tensor([]), query, self.obj_classes[query], presence


class FoVTask4(FoVTask3):

  def __init__(self, splits, direction,
               data_root='../data/fov_pretraining',
               images='all',
               obj_dict_file='../data/vg_object_dictionaries.json',
               use_objects=False,
               obj_complement='pano',
               ignore_list=[]):
    super(FoVTask4, self).__init__(splits, direction,
                                   data_root=data_root,
                                   images=images,
                                   obj_dict_file=obj_dict_file,
                                   use_objects=use_objects,
                                   ignore_list=ignore_list)

  def __getitem__(self, index):

    obj_query = self.obj_queries[index]
    pano_id, pano_category, pano_loc = self.pano_metas[index]

    existing_panos = set(self.obj2pano[obj_query])
    all_panos = set(self.pano2fov.keys())
    not_present = all_panos.difference(existing_panos)

    if random.randint(0, 1) == 1 and len(not_present):
      fov_rnd = random.choice(list(not_present))
      index = self.pano2idx[fov_rnd]
      presence = torch.cuda.FloatTensor([0])
    else:
      presence = torch.cuda.FloatTensor([1])

    (lat, lng) = self.fovs[index]
    latitude = torch.cuda.FloatTensor([lat])
    longitude = torch.cuda.FloatTensor([lng])

    fov_file = self.fov_files[index]

    img_feat_path = fov_file.replace('.jpg', '.feat.npy')
    img_features = np.load(img_feat_path)

    image = torch.cuda.FloatTensor(img_features)
    query = torch.cuda.LongTensor([obj_query])

    if self.use_objects:
      img_id = fov_file.split('/')[-1].replace('.jpg', '')
      img_info = self.imgid2img[img_id]

      boxes = torch.cuda.FloatTensor(img_info['boxes'].copy())
      feats = torch.cuda.FloatTensor(img_info['features'].copy())

      img_h, img_w = img_info['img_h'], img_info['img_w']
      boxes[..., (0, 2)] /= img_w
      boxes[..., (1, 3)] /= img_h

      return latitude, longitude, image, boxes, feats, query, self.obj_classes[query], presence
    return latitude, longitude, image, torch.tensor([]), torch.tensor([]), query, self.obj_classes[query], presence

  def __len__(self):
    return len(self.pano_metas)
