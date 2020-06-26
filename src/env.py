'''
Batched refer360 environment
The main structure is similar to https://github.com/peteanderson80/Matterport3DSimulator/blob/master/tasks/R2R/env.py
'''
from dict import Dictionary
from utils import load_datasets
from utils import smoothed_gaussian
from utils import gaussian_target
from utils import rad2degree
from panoramic_camera import PanoramicCamera as camera
from torchvision import transforms
import numpy as np
import torch
from collections import defaultdict
from PIL import Image
EPS = 1e-10
FOV_SIZE = 400


class EnvBatch():
  '''Wrapper for a batch of cameras.
  '''

  def __init__(self, batch_size=64,
               image_h=FOV_SIZE,
               image_w=FOV_SIZE,
               fov=90,
               degrees=5):
    '''Initialize camerase for a new batch of environment.
    '''

    self.degrees = degrees
    self.__look__ = {1: (0, self.degrees),
                     2: (0, -self.degrees),
                     3: (self.degrees, 0),
                     4: (-self.degrees, 0),
                     }

    self.batch_size = batch_size
    self.image_w = image_w
    self.image_h = image_h
    self.fov = 90

    shape = (image_h, image_w)
    # set # of cameras batch_size
    self.cameras = [camera(fov=fov,
                           output_image_shape=shape) for ii in range(self.batch_size)]
    # keep track of which cameras are done
    self._done = [False for ii in range(self.batch_size)]

  def newEpisodes(self, pano_paths, gt_loc=[]):
    '''Initialize cameras with new images.
    '''
    for ii, pano_path in enumerate(pano_paths):
      self.cameras[ii].load_img(
          pano_path, gt_loc=gt_loc[ii])
      self._done[ii] = False

  def getStates(self):
    '''Return (image, pixel_map, done) tuple
    '''
    states = []
    for ii in range(self.batch_size):
      img = self.cameras[ii].get_image()
      pixel_map = self.cameras[ii].get_map()
      done = self._done[ii]
      states.append((img, pixel_map, done))

    return states

  def makeActions(self, action_tuples):
    '''Execute action tuples for each camera.
    '''

    for ii, action in enumerate(action_tuples):

      navigate, argmax_pred, pred = action
      navigate = navigate.item()

      if not self._done[ii]:
        if navigate in [1, 2, 3, 4]:
          longitude = self.cameras[ii].lng + self.__look__[navigate][1]
          latitude = self.cameras[ii].lat + self.__look__[navigate][0]

          # if long < -90 or long > 90 lat switches to other side
          longitude = min(max(-90, longitude), 90)
          if longitude < -90:
            longitude = -90 - (longitude + 90)
            latitude = latitude + 180
          elif longitude > 90:
            longitude = 90 - (longitude - 90)
            latitude = latitude + 180

          if latitude > 180:
            latitude = latitude - 360
          elif latitude < -180:
            latitude = latitude + 360

          self.cameras[ii].lng = longitude
          self.cameras[ii].lat = latitude

          self.cameras[ii].look(latitude, longitude)


class Refer360Batch():
  '''Refer360 environment
  '''

  def __init__(self, splits=['train'], batch_size=64, seed=0,
               image_h=FOV_SIZE,
               image_w=FOV_SIZE,
               fov=90,
               degrees=5,
               use_sentences=False,
               use_gt_action=False,
               oracle_mode=False,
               prepare_vocab=False,
               images='all'):

    self.batch_size = batch_size
    self.image_w = image_w
    self.image_h = image_h
    self.fov = 90
    self.degrees = degrees

    # observed image size
    self.observation_space = (3, FOV_SIZE, FOV_SIZE)
    self.action_space = 5
    self.images = images

    # set the number of cameras(envs) to batch_size
    self.env = EnvBatch(batch_size=self.batch_size,
                        image_h=self.image_h,
                        image_w=self.image_w,
                        fov=self.fov,
                        degrees=self.degrees)
    # keep a list of datum
    self.data = []

    # load splits
    self.vocab = None

    connectivity = []
    for ii, split_name in enumerate(splits):
      data_split, sentences = load_datasets([split_name], self.images)
      for jj, datum in enumerate(data_split[0]):
        datum['id'] = '{}_{}'.format(split_name, jj)

        if 'nodes' in datum:
          for node in datum['nodes']:
            connectivity += [len(datum['nodes'][node]['neighbors'])]
        self.data.append(datum)

      if prepare_vocab:
        self.vocab = Dictionary(sentences, min_freq=3, split=True)
    print('{} split(s) loaded with {} instances'.format(
        ','.join(splits), len(self.data)))

    if 'nodes' in datum:
      print('min mean max connectivity %d %d %d' %
            (np.min(connectivity), np.mean(connectivity), np.max(connectivity)))
      self.max_connectivity = np.max(connectivity)

    else:
      self.max_connectivity = 0
    self.use_sentences = use_sentences
    self.use_gt_action = use_gt_action
    if self.use_sentences:
      self.sentence_ids = [0 for ii in range(self.batch_size)]
    self.oracle_mode = oracle_mode
    self.diff_threshold = 20
    self.coor_threshold = 20
    self.preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    self.predictions = [None for ii in range(self.batch_size)]

    np.random.seed(seed)
    torch.manual_seed(seed)

  def __len__(self):
    return len(self.data)

  def _next_minibatch(self):
    '''Set the new minibatch
    '''
    batch = self.data[self.ix:self.ix+self.batch_size]
    if len(batch) < self.batch_size:
      np.random.shuffle(self.data)
      self.ix = self.batch_size - len(batch)
      batch += self.data[:self.ix]
    else:
      self.ix += self.batch_size
    self.batch = batch
    self._reset_predictions()

  def _get_gt_action(self, idx):
    '''Returns shortest-path action towards the target location.
    '''
    datum = self.batch[idx]

    # shifting from -180,180 to 0,360
    latitude = self.env.cameras[idx].lat + 180
    longitude = self.env.cameras[idx].lng

    # find shortest-path action
    gt_lng = datum['gt_lng']
    gt_lat = datum['gt_lat']
    lng_diff = 0
    lat_diff = 0

    pred, argmax_pred = torch.tensor([]).cuda(), torch.tensor([]).cuda()
    lng_diff = gt_lng - longitude
    coor = None

    gt_lat = datum['gt_lat'] + 180
    gt_lng = datum['gt_lng']

    if gt_lat < latitude:
      dist1 = latitude - gt_lat  # r->l distance
      dist2 = gt_lat + 360 - latitude  # l->r distance
    else:
      dist2 = gt_lat - latitude  # l->r
      dist1 = latitude + 360 - gt_lat  # r->l
    if dist2 < dist1:  # move right
      gt_action = 3
      lat_diff = dist2
    else:  # move left
      gt_action = 4
      lat_diff = dist1
    if np.abs(lng_diff) > lat_diff:
      gt_action = 1 if lng_diff > 0 else 2

    if np.abs(lat_diff) + np.abs(lng_diff) < self.diff_threshold:
      coor = self.env.cameras[idx].get_image_coordinate_for(
          datum['gt_lat'], datum['gt_lng'])
      if coor and self.coor_threshold <= coor[1] <= FOV_SIZE-self.coor_threshold and self.coor_threshold <= coor[0] <= FOV_SIZE-self.coor_threshold:
        # FIX FoV=400 but pred FoV is 100
        x, y = int(coor[1]/4), int(coor[0]/4)
        pred = gaussian_target(x, y, height=100, width=100)
        argmax_pred = torch.tensor([x, y]).cuda()
        gt_action = 0

    gt_tuple = (torch.tensor(gt_action), argmax_pred, pred)
    return gt_tuple, coor

  def _get_obs(self):
    '''Returns a list of information and meta-data
    '''
    # keep a list of metadata
    infos = []
    # keep a list of image observations
    im_stack = []
    for ii, state in enumerate(self.env.getStates()):
      datum = self.batch[ii]

      # Check dimension order
      perspective_image = state[0].astype('uint8')
      pil_img = Image.fromarray(perspective_image)
      img_tensor = self.preprocess(pil_img)
      im_stack.append(img_tensor.unsqueeze(0))
      # im_stack.append(torch.from_numpy(
      #     np.expand_dims(state[0], axis=0)).permute(0, 3, 1, 2).float())

      if self.use_sentences:
        refexps = [datum['refexps'][self.sentence_ids[ii]]]
      else:
        refexps = datum['refexps']

      if self.use_gt_action:
        gt_tuple, gt_coor = self._get_gt_action(ii)
      else:
        gt_tuple, gt_coor = (None, None, None), None

      infos.append({
          'id': datum['annotationid'],
          'gt_moves': datum['gt_moves'][0],
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
    im_batch = torch.cat(im_stack, 0).cuda()
    return im_batch, infos

  def reset_panos(self):
    '''Random start lat/lng
    '''
    for ii in range(self.batch_size):
      slat, slng = rad2degree(np.random.uniform(0, 6),
                              np.random.uniform(1, 1.5))

      self.env.cameras[ii].lng = slng
      self.env.cameras[ii].lat = slat
      self.env.cameras[ii].look(
          self.env.cameras[ii].lat, self.env.cameras[ii].lng)

  def reset_epoch(self):
    '''Resets the epoch.
    '''
    self.ix = 0

  def next(self, width=4552, height=2276):
    '''Set new batch.
    '''
    self._next_minibatch()

    pano_paths = [datum['pano'] for datum in self.batch]
    self.sentence_max = [len(datum['gt_moves'][0]) for datum in self.batch]
    self.sentence_ids = [0 for ii in range(self.batch_size)]
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

  def _reset_predictions(self):
    '''Reset the predictions
    '''
    self.predictions = [None for ii in range(self.batch_size)]

  def nextSentence(self):
    '''Retrurn the next sentence.
    '''
    for ii in range(self.batch_size):
      if self.sentence_ids[ii] < self.sentence_max[ii] - 1:
        self.sentence_ids[ii] += 1
        self.env._done[ii] = False

  def nextStartPoint(self):
    '''Jump to the next ground-truth location.
    '''
    for ii in range(self.batch_size):
      n_sentence = self.sentence_ids[ii]
      self.env.cameras[ii].lng = self.batch[ii]['gt_moves'][0][n_sentence][1]
      self.env.cameras[ii].lat = self.batch[ii]['gt_moves'][0][n_sentence][0]
      self.env.cameras[ii].look(
          self.env.cameras[ii].lat, self.env.cameras[ii].lng)
    obs, infos = self._get_obs()
    return obs, infos

  def eval_batch(self):
    '''Evaluate the batch.
    '''
    metrics = defaultdict(float)
    for ii in range(self.batch_size):
      if self.predictions[ii]:
        pred = self.predictions[ii]
        distance = np.sqrt(((pred[0][0] - pred[1][0])
                            ** 2 + (pred[0][1] - pred[1][1])**2).item())
        for th in [40, 80, 120]:
          metrics['acc_{}'.format(th)] += int(distance <= th)
        metrics['fov'] += 1.0
        metrics['distance'] += distance
      else:
        metrics['distance'] += (100*np.sqrt(2))
        for th in [40, 80, 120]:
          metrics['acc_{}'.format(th)] += 0
        metrics['fov'] += 0.0

    for metric in metrics:
      metrics[metric] = metrics[metric]/self.batch_size
    return metrics

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
