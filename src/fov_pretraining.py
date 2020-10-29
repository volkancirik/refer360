from torch.utils.data import Dataset
import numpy as np
import os
import torch


def load_fovpretraining_splits(splits,
                               data_root='../data/fov_pretraining',
                               images='all'):

  fov_files = []
  regions = []
  refexps = []

  for split in splits:
    split_file = os.path.join(
        data_root, '{}.[{}].imdb.npy'.format(split, images))
    print('loading split {} from {} for FoV pretraining'.format(split, split_file))
    instances = np.load(split_file, allow_pickle=True)[()]['data_list'][0]
    for instance in instances:
      fov_files += [instance['fov_file']]
      regions += [instance['regions']]
      refexps += [instance['refexp']]
  return fov_files, regions, refexps


class FoVPretrainingDataset(Dataset):
  def __init__(self, splits,
               data_root='../data/fov_pretraining',
               images='all'):

    fov_files, regions, refexps = load_fovpretraining_splits(splits,
                                                             data_root=data_root,
                                                             images=images)
    self.fov_files = fov_files
    self.regions = regions
    self.refexps = refexps

  def __getitem__(self, index):
    fov_file = self.fov_files[index]
    regions = self.regions[index]

    img_feat_path = fov_file.replace('.jpg', '.feat.npy')
    img_features = np.load(img_feat_path)

    image = torch.cuda.FloatTensor(img_features)
    target_up = torch.cuda.FloatTensor(regions['up']).resize(1, 1600)
    target_down = torch.cuda.FloatTensor(regions['down']).resize(1, 1600)
    target_left = torch.cuda.FloatTensor(regions['left']).resize(1, 1600)
    target_right = torch.cuda.FloatTensor(regions['right']).resize(1, 1600)

    target = torch.cat([target_up,
                        target_down,
                        target_left,
                        target_right], axis=1).squeeze(0)
    return image, target

  def __len__(self):
    return len(self.regions)
