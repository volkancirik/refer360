from collections import defaultdict
from torch.utils.data import Dataset
from utils import DIRECTIONS, CAT2LOC
from utils import get_object_dictionaries
from model_utils import load_obj_tsv
from operator import itemgetter
import numpy as np
import os
import torch
import random
from tqdm import tqdm
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_fovpretraining_splits(splits,
                               direction,
                               data_root='../data/fov_pretraining',
                               images='all',
                               task='task1',
                               obj_classes=[],
                               ignore_list='',
                               use_meta=False):

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
      if use_meta:
        pano_category = instance['pano'].split('/')[-2]
        pano_id = "_".join(instance['pano'].split(
            '/')[-1].split('.')[0].split('_')[:2])
        pano_loc = CAT2LOC[pano_category]
      else:
        pano_category = 'n/a'
        pano_id = instance['pano']
        pano_loc = 'n/a'

      pano_metas += [(pano_id, pano_category, pano_loc)]
      fovs += [(instance['latitude'], instance['longitude'])]
      try:
        refexps += [instance['refexp']]
      except:
        refexps += [instance['refexps']]
        pass

      regions += [instance['regions']]
      obj_lists += [instance['obj_list']]

      obj_query = []
      obj_direction = []
      for dir_id, dir in enumerate(DIRECTIONS[direction]):
        for obj in instance['obj_list'][direction][dir]:
          obj_query += [obj]
          obj_direction += [dir_id]

          if obj not in labels:
            labels[obj] = defaultdict(int)
          labels[obj][dir] += 1.0
      obj_queries += [obj_query]
      obj_directions += [obj_direction]
    pbar.close()
  print('fovs:', len(fovs))
  print('fov_files:', len(fov_files))
  print('regions:', len(regions))
  print('refexps:', len(refexps))
  print('obj_lists:', len(obj_lists))
  print('obj_queries:', len(obj_queries))
  print('obj_directions:', len(obj_directions))
  print('labels:', len(labels))
  print('pano_metas:', len(pano_metas))

  most_freq = defaultdict(int)
  if 'train' in splits:
    print('_'*20)
    for o, _ in enumerate(obj_classes):
      sorted_counts = sorted(labels[o].items(), key=itemgetter(1))[::-1]
      total = sum([cnt for _, cnt in sorted_counts])
      print('object:', obj_classes[o], 'total count:', int(total))
      for ii, (d, cnt) in enumerate(sorted_counts):
        print('{:20}:{:3.3f}'.format(d, cnt/total))
      print('_'*20)
      most_freq[o] = DIRECTIONS[direction].index(sorted_counts[0][0])
    print('most_freq:', most_freq)
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

    vg2idx, idx2vg, obj_classes = get_object_dictionaries(obj_dict_file)
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
    image = torch.FloatTensor(img_features).to(DEVICE)
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
    latitude = torch.FloatTensor([lat]).to(DEVICE)
    longitude = torch.FloatTensor([lng]).to(DEVICE)

    fov_file = self.fov_files[index]
    regions = self.regions[index]

    img_feat_path = fov_file.replace('.jpg', '.feat.npy')
    img_features = np.load(img_feat_path)

    n_objects = len(self.vg2idx)

    image = torch.FloatTensor(img_features).to(DEVICE)

    targets = []
    for dir in DIRECTIONS[self.direction]:
      targets.append(torch.FloatTensor(
          regions[self.direction][dir]).resize(1, n_objects).to(DEVICE))
    target = torch.cat(targets,
                       axis=1).squeeze(0)

    if self.use_objects:
      img_id = fov_file.split('/')[-1].replace('.jpg', '')
      img_info = self.imgid2img[img_id]

      boxes = torch.FloatTensor(img_info['boxes'].copy()).to(DEVICE)
      feats = torch.FloatTensor(img_info['features'].copy()).to(DEVICE)

      img_h, img_w = img_info['img_h'], img_info['img_w']
      boxes[..., (0, 2)] /= img_w
      boxes[..., (1, 3)] /= img_h

      return latitude, longitude, image, boxes, feats, torch.tensor([]), '', target, torch.tensor([])
    return latitude, longitude, image, torch.tensor([]), torch.tensor([]), torch.tensor([]), '', target, torch.tensor([])


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

    latitude = torch.FloatTensor([lat]).to(DEVICE)
    longitude = torch.FloatTensor([lng]).to(DEVICE)

    fov_file = self.fov_files[index]

    obj_queries = self.obj_queries[index]
    obj_directions = self.obj_directions[index]

    obj_index = random.randint(0, len(obj_queries)-1)
    obj_query = obj_queries[obj_index]
    obj_direction = obj_directions[obj_index]
    most_freq = self.most_freq[obj_query]

    img_feat_path = fov_file.replace('.jpg', '.feat.npy')
    img_features = np.load(img_feat_path)

    image = torch.FloatTensor(img_features).to(DEVICE)
    query = torch.LongTensor([obj_query]).to(DEVICE)
    direction = torch.FloatTensor([obj_direction]).to(DEVICE)
    most_freq_label = torch.LongTensor([most_freq]).to(DEVICE)

    if self.use_objects:
      img_id = fov_file.split('/')[-1].replace('.jpg', '')
      img_info = self.imgid2img[img_id]

      boxes = torch.FloatTensor(img_info['boxes'].copy()).to(DEVICE)
      feats = torch.FloatTensor(img_info['features'].copy()).to(DEVICE)

      img_h, img_w = img_info['img_h'], img_info['img_w']
      boxes[..., (0, 2)] /= img_w
      boxes[..., (1, 3)] /= img_h

      return latitude, longitude, image, boxes, feats, query, self.obj_classes[query], direction, most_freq_label
    return latitude, longitude, image, torch.tensor([]), torch.tensor([]), query, self.obj_classes[query], direction, most_freq_label


class FoVTask3(FoVPretrainingDataset):

  def __init__(self, splits, direction,
               data_root='../data/fov_pretraining',
               images='all',
               obj_dict_file='../data/vg_object_dictionaries.json',
               use_objects=False,
               obj_complement='pano',
               ignore_list='',
               task='task3'):
    super(FoVTask3, self).__init__(splits, direction,
                                   data_root=data_root,
                                   images=images,
                                   task=task,
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
    latitude = torch.FloatTensor([lat]).to(DEVICE)
    longitude = torch.FloatTensor([lng]).to(DEVICE)

    fov_file = self.fov_files[index]

    obj_queries = self.obj_queries[index]
    obj_directions = self.obj_directions[index]

    obj_index = random.randint(0, len(obj_queries)-1)
    obj_query = obj_queries[obj_index]
    obj_direction = obj_directions[obj_index]
    most_freq = self.most_freq[obj_query]

    img_feat_path = fov_file.replace('.jpg', '.feat.npy')
    img_features = np.load(img_feat_path)

    image = torch.FloatTensor(img_features).to(DEVICE)
    query = torch.LongTensor([obj_query]).to(DEVICE)
    most_freq_label = torch.LongTensor([most_freq]).to(DEVICE)

    pano_id, pano_category, pano_loc = self.pano_metas[index]

    diff_set = list(set([int(key) for key in self.idx2vg.keys()]
                        ).difference(self.fov2obj[fov_file]))
    if random.randint(0, 1) == 1 and len(diff_set) > 0:
      obj_query = random.choice(diff_set)
      query = torch.LongTensor([obj_query]).to(DEVICE)
      presence = torch.FloatTensor([0]).to(DEVICE)
    else:
      presence = torch.FloatTensor([1]).to(DEVICE)

    if self.use_objects:
      img_id = fov_file.split('/')[-1].replace('.jpg', '')
      img_info = self.imgid2img[img_id]

      boxes = torch.FloatTensor(img_info['boxes'].copy()).to(DEVICE)
      feats = torch.FloatTensor(img_info['features'].copy()).to(DEVICE)

      img_h, img_w = img_info['img_h'], img_info['img_w']
      boxes[..., (0, 2)] /= img_w
      boxes[..., (1, 3)] /= img_h

      return latitude, longitude, image, boxes, feats, query, self.obj_classes[query], presence, most_freq_label
    return latitude, longitude, image, torch.tensor([]), torch.tensor([]), query, self.obj_classes[query], presence, most_freq_label


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
                                   ignore_list=ignore_list,
                                   task='task4')

  def __getitem__(self, index):

    obj_queries = self.obj_queries[index]
    #obj_directions = self.obj_directions[index]

    obj_index = random.randint(0, len(obj_queries)-1)
    obj_query = obj_queries[obj_index]
    #obj_direction = obj_directions[obj_index]
    most_freq = self.most_freq[obj_query]

    pano_id, pano_category, pano_loc = self.pano_metas[index]

    existing_panos = set(self.obj2pano[obj_query])
    all_panos = set(self.pano2fov.keys())
    not_present = all_panos.difference(existing_panos)

    if random.randint(0, 1) == 1 and len(not_present):
      fov_rnd = random.choice(list(not_present))
      index = self.pano2idx[fov_rnd]
      presence = torch.FloatTensor([0]).to(DEVICE)
    else:
      presence = torch.FloatTensor([1]).to(DEVICE)

    (lat, lng) = self.fovs[index]
    latitude = torch.FloatTensor([lat]).to(DEVICE)
    longitude = torch.FloatTensor([lng]).to(DEVICE)

    fov_file = self.fov_files[index]

    img_feat_path = fov_file.replace('.jpg', '.feat.npy')
    img_features = np.load(img_feat_path)

    image = torch.FloatTensor(img_features).to(DEVICE)
    query = torch.LongTensor([obj_query]).to(DEVICE)
    most_freq_label = torch.LongTensor([most_freq]).to(DEVICE)

    if self.use_objects:
      img_id = fov_file.split('/')[-1].replace('.jpg', '')
      img_info = self.imgid2img[img_id]

      boxes = torch.FloatTensor(img_info['boxes'].copy()).to(DEVICE)
      feats = torch.FloatTensor(img_info['features'].copy()).to(DEVICE)

      img_h, img_w = img_info['img_h'], img_info['img_w']
      boxes[..., (0, 2)] /= img_w
      boxes[..., (1, 3)] /= img_h

      return latitude, longitude, image, boxes, feats, query, self.obj_classes[query], presence, most_freq_label
    return latitude, longitude, image, torch.tensor([]), torch.tensor([]), query, self.obj_classes[query], presence, most_freq_label

  def __len__(self):
    return len(self.pano_metas)
