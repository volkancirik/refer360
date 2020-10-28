'''
Batched refer360 environment
'''
import cv2
from dict import Dictionary
from utils import load_datasets
from utils import smoothed_gaussian
from utils import gaussian_target
from panoramic_camera import PanoramicCamera as camera
from torchvision import transforms
import numpy as np
import torch
from collections import defaultdict
EPS = 1e-10
FOV_SIZE = 512

from env import EnvBatch
from env import Refer360Batch


'''
navigation actions:
0 : stop and predict location in the FoV
1 : next sentence
2 : change the FoV
'''


class GraphEnvBatch(EnvBatch):
  '''Wrapper for a batch of cameras.
  '''

  def __init__(self, batch_size=64,
               image_h=FOV_SIZE,
               image_w=FOV_SIZE,
               fov=90,
               degrees=5):
    '''Initialize camerase for a new batch of environment.
    '''
    self.batch_size = batch_size
    self.image_w = image_w
    self.image_h = image_h
    self.fov = 90

    shape = (image_h, image_w)
    # set # of cameras batch_size
    self._cameras = [camera(fov=fov,
                            output_image_shape=shape) for ii in range(self.batch_size)]
    self._fov_ids = [0 for ii in range(self.batch_size)]
    self._fovs = [None for ii in range(self.batch_size)]

    self.done = [False for ii in range(self.batch_size)]
    self.fov_prefixes = [None for ii in range(self.batch_size)]

  def changeFov(self, idx, fov_id):
    self._fov_ids[idx] = fov_id
    self._fov = cv2.imread(
        self.fov_prefixes[idx] + '{}.jpg'.format(idx), cv2.IMREAD_COLOR)

  def newEpisodes(self, pano_paths, fov_prefixes, gt_loc=[]):
    '''Initialize cameras with new images.
    '''
    for ii, (pano_path, fov_prefix) in enumerate(zip(pano_paths, fov_prefixes)):

      self._cameras[ii].load_img(
          pano_path, gt_loc=gt_loc[ii])

      self.done[ii] = False
      self.fov_prefixes[ii] = fov_prefix
      self.changeFov(ii, 0)

  def makeActions(self, action_tuples):
    '''Execute action tuples for each camera.
    action_tuple = navigate, new_fov, argmax_pred, pred

    navigate = torch.tensor of size 1 in [0,1,2]
    new_fov = node_id for the next FoV
    argmax_pred = (x,y) prediction
    pred = torch.tensor of size fov_height x fov_width
    '''

    for ii, action in enumerate(action_tuples):

      navigate, new_fov, _, _ = action
      navigate = navigate.item()

      if not self._done[ii]:
        if navigate == 0:
          # stop and predict
          continue
        elif navigate == 1:
          # the next sentence
          continue
        elif navigate == 2:
          # change the FoV
          self.changeFov(ii, new_fov)
        else:
          raise NotImplementedError('navigate not in [0,1,2]')


class Refer360GraphBatch(Refer360Batch):
  '''Refer360 environment
  '''

  def _get_gt_action(self, idx):
    '''Returns ground-truth action tuple for datum at index idx.
    '''
    datum = self.batch[idx]

    # default values
    pred, argmax_pred = torch.tensor([]).cuda(), torch.tensor([]).cuda()
    new_fov_id = -1
    coor = None

    hop_id = self.hop_ids[idx]
    sen_id = self.sentence_ids[idx]

    # hops are list of lists for each datum
    # each sentence has a list of hops
    max_sen = len(datum['gt_hops'])
    max_hop = len(datum['gt_hops'][sen_id])

    gt_lng = datum['gt_lng']
    gt_lat = datum['gt_lat']

    # last sentence last hop --> predict x,y
    if hop_id == max_hop and sen_id == max_sen-1:
      coor = self.env.cameras[idx].get_image_coordinate_for(gt_lat, gt_lng)
      if coor and self.coor_threshold <= coor[1] <= FOV_SIZE-self.coor_threshold and self.coor_threshold <= coor[0] <= FOV_SIZE-self.coor_threshold:
        x, y = int(coor[1]), int(coor[0])
        pred = gaussian_target(x, y, height=FOV_SIZE, width=FOV_SIZE)
        argmax_pred = torch.tensor([x, y]).cuda()
        gt_action = 0  # stop and predict
    else:
      if hop_id == max_hop:  # new sentence
        gt_action = 1
      elif hop_id < max_hop:  # change FoV
        gt_action = 2
        # idx of next FoV among neighbors
        new_fov_id = datum['gt_hops'][sen_id][hop_id]
      else:
        raise NotImplementedError('hop_id > max_hop')
    gt_tuple = (torch.tensor(gt_action), torch.tensor(
        new_fov_id), argmax_pred, pred)
    return gt_tuple, coor

  def _get_obs(self):
    '''Return a list of observations for each datum in the batch.
    '''
    # keep a list of metadata
    infos = []
    # keep a list of image observations
    cur_im_stack = []
    next_im_stack = []

    for ii, state in enumerate(self.env.getStates()):
      datum = self.batch[ii]

      # Check dimension order
      img_tensor = self.preprocess(state[0])
      cur_im_stack.append(img_tensor.unsqueeze(0))

      current_node = self.current_node[ii]

      cur_fov_stack = []
      ###
      # for n in self.datum['nodes'][current_node]['neighbors']:

      refexps = [datum['refexps'][self.sentence_ids[ii]]]

      if self.use_gt_action:
        gt_tuple, gt_coor = self._get_gt_action(ii)
      else:
        gt_tuple, gt_coor = (None, None, None, None), None

      infos.append({
          'id': datum['annotationid'],
          'refexps': refexps,
          'img_category': datum['img_category'],
          'gt_lat': datum['gt_lat'],
          'gt_lng': datum['gt_lng'],
          'gt_tuple': gt_tuple,
          'gt_coor': gt_coor,
          'pano': datum['pano'],
          'img': state[0],
          'pixel_map': state[1],
          'latitude': self.env.cameras[ii].lat,
          'longitude': self.env.cameras[ii].lng,
      })
    cur_im_batch = torch.cat(cur_im_stack, 0).cuda()
    next_im_batch = None
    return cur_im_batch, next_im_batch, infos

  def reset_panos(self):
    '''Random start lat/lng
    '''
    for ii in range(self.batch_size):
      datum = self.batch[ii]
      self.env.cameras[ii].lng = datum['nodes'][0]['lng']
      self.env.cameras[ii].lat = datum['nodes'][0]['lat']
      self.env.cameras[ii].look(
          self.env.cameras[ii].lat, self.env.cameras[ii].lng)

  def next(self, width=4552, height=2276):
    '''Set new batch.
    '''
    self._next_minibatch()

    pano_paths = [datum['pano'] for datum in self.batch]
    self.sentence_max = [len(datum['gt_hops']) for datum in self.batch]
    self.sentence_ids = [0 for ii in range(self.batch_size)]
    self.hop_ids = [0 for ii in range(self.batch_size)]
    if self.oracle_mode:
      gt_loc = []
      for ii, datum in enumerate(self.batch):
        latitude = datum['gt_lat']
        longitude = datum['gt_lng']
        gt_x = int(width * ((latitude)/360.0))
        gt_y = height - int(height * ((longitude + 90)/180.0))
        gt_loc.append((gt_y, gt_x))
    else:
      gt_loc = [None]*self.batch_size

    self.env.newEpisodes(pano_paths, gt_loc=gt_loc)
    self.reset_panos()
    obs, infos = self._get_obs()
    return obs, infos

  def nextStartPoint(self):
    for ii in range(self.batch_size):
      n_sentence = self.sentence_ids[ii]
      nid = self.batch[ii]['gt_hops'][n_sentence][0]
      self.env.cameras[ii].lng = self.batch[ii]['nodes'][nid]['lng']
      self.env.cameras[ii].lat = self.batch[ii]['nodes'][nid]['lng']
      self.env.cameras[ii].look(
          self.env.cameras[ii].lat, self.env.cameras[ii].lng)
    obs, infos = self._get_obs()
    return obs, infos

  def step(self, action_tuples, life_penalty=-1):
    '''Make one step for each batch item
    '''
    self.env.makeActions(action_tuples)

    reward = [life_penalty]*self.batch_size
    obs, infos = self._get_obs()
    for ii, action in enumerate(action_tuples):
      navigate, argmax_pred, pred = action

     # for unfinished environment set the reward
      if not self.env._done[ii]:
        datum = self.batch[ii]
        coor = self.env.cameras[ii].get_image_coordinate_for(
            datum['gt_lat'], datum['gt_lng'])

        if coor:  # waldo is in the fov
          # if the agent makes a prediction
          if navigate == 0:
            # return negative of kl_loss as reward
            gt_x, gt_y = int(coor[1]/4), int(coor[0]/4)
            kl_loss = smoothed_gaussian(pred, gt_x, gt_y,
                                        height=100, width=100)
            kl_loss = kl_loss.item()
            reward[ii] = (-kl_loss)*10
            self.env._done[ii] = True
            self.predictions[ii] = (argmax_pred, (gt_x, gt_y))
          else:
            # give 0 reward since waldo in fov
            reward[ii] = 0.
        elif navigate == 0:
          # waldo is not in fov
          reward[ii] = life_penalty*2
          self.env._done[ii] = True
        elif navigate == 5 and self.use_sentences:
          n_sentence = self.sentence_ids[ii]
          target_lng = self.batch[ii]['gt_moves'][0][n_sentence][1]
          target_lat = self.batch[ii]['gt_moves'][0][n_sentence][0]
          target_coor = self.env.cameras[ii].get_image_coordinate_for(
              target_lat, target_lng)
          if target_coor:
            reward[ii] = 300 - ((target_coor[0] - 200) **
                                2 + (target_coor[1] - 200)**2)*0.5
            print('\n, target_coor', target_coor, reward[ii])
          else:
            reward[ii] = life_penalty*2
          self.env._done[ii] = True

    reward = torch.from_numpy(np.array([reward])).permute(1, 0).float()
    return obs, reward, np.array(self.env._done), infos
