import paths

import numpy as np
import torchvision.models as models
import torch
from torchvision import transforms
from tqdm import tqdm
from fov_pretraining import load_fovpretraining_splits
from PIL import Image
import sys
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def compute_image_caches(splits,
                         task='',
                         data_root='',
                         images='all',
                         extension='.jpg'):

  if task == 'fov_pretraining':
    out = load_fovpretraining_splits(splits,
                                     'cartesian',
                                     data_root=data_root,
                                     images=images)
    files = out[1]
  else:
    raise NotImplementedError()

  resnetmodel = models.resnet152(pretrained=True).to(DEVICE)
  newmodel = torch.nn.Sequential(
      *(list(resnetmodel.children())[:-2])).to(DEVICE)
  for p in newmodel.parameters():
    newmodel.requires_grad = False
  preprocess = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
  ])

  print('# of files {} # of unique files {}'.format(len(files), len(set(files))))
  pbar = tqdm(list(set(files)))
  for filename in pbar:
    img_feat_path = filename.replace(extension, '.feat.npy')

    feats = []

    image_obj = Image.open(filename)
    img_tensor = preprocess(image_obj)
    feats = newmodel(img_tensor.to(DEVICE).unsqueeze(0)).squeeze(0)
    feats = feats.cpu().detach().numpy()

    np.save(img_feat_path, feats)
  pbar.close()


if __name__ == '__main__':
  from utils import SPLITS
  compute_image_caches(['train', 'val_seen', 'val_unseen', 'test'],
                       task='fov_pretraining',
                       data_root=sys.argv[1],
                       images='all')
