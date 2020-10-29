import cv2
import numpy as np
import torchvision.models as models
import torch
from torchvision import transforms
from tqdm import tqdm
from fov_pretraining import load_fovpretraining_splits
from PIL import Image


def compute_image_caches(splits,
                         task='fov_pretraining',
                         data_root='../data/fov_pretraining',
                         images='all',
                         extension='.jpg'):

  if task == 'fov_pretraining':
    files, _, _ = load_fovpretraining_splits(splits,
                                             data_root=data_root,
                                             images=images)
  else:
    raise NotImplementedError()

  resnetmodel = models.resnet18(pretrained=True).cuda()
  newmodel = torch.nn.Sequential(
      *(list(resnetmodel.children())[:-2])).cuda()
  for p in newmodel.parameters():
    newmodel.requires_grad = False
  preprocess = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])

  pbar = tqdm(files)
  for filename in pbar:
    img_feat_path = filename.replace(extension, '.feat.npy')

    feats = []

    image_obj = Image.open(filename)
    image_obj = cv2.cvtColor(
        np.array(image_obj), cv2.COLOR_RGB2BGR).astype('uint8')
    img_tensor = preprocess(image_obj)

    feats = newmodel(img_tensor.cuda().unsqueeze(0)).squeeze(0)
    feats = feats.cpu().detach().numpy()

    np.save(img_feat_path, feats)
  pbar.close()


if __name__ == '__main__':
  from utils import SPLITS
  compute_image_caches(SPLITS,
                       task='fov_pretraining',
                       data_root='../data/fov_pretraining',
                       images='all')
