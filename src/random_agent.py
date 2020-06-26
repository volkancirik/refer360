"""
Random agent playing the game
"""
from env import Refer360Batch
import json

import numpy as np
import os
import torch

FOV_SIZE = 100
DEGREES = 15
__look__ = {1: (0, DEGREES),
            2: (0, -DEGREES),
            3: (DEGREES, 0),
            4: (-DEGREES, 0),
            }


def dump_history(h, hid):

  path = './exp-random/samples/'
  os.makedirs(path, exist_ok=True)
  with open(os.path.join(path, '{}'.format(hid) + "_randomagent.json"), 'w') as outfile:
    json.dump(h, outfile)


def predict_location(o, history):
  pred = np.random.uniform(-0.99, -0.0001, size=(FOV_SIZE, FOV_SIZE))
  pred_x, pred_y = np.unravel_index(pred.argmax(), pred.shape)

  pixel_map = o['pixel_map']

  pred_lat = pixel_map[pred_y][pred_x][0]
  pred_lng = pixel_map[pred_y][pred_x][1]

  history[o['id']]['pred_x'] = int(pred_x)
  history[o['id']]['pred_y'] = int(pred_y)

  history[o['id']]['pred_lat'] = float(pred_lat)
  history[o['id']]['pred_lng'] = float(pred_lng)

  history[o['id']]['gt_lat'] = float(o['gt_lat'])
  history[o['id']]['gt_lng'] = float(o['gt_lng'])
  history[o['id']]['refexps'] = o['refexps']

  dump_history(history[o['id']], o['id'])
  return pred, float(pred_lat), float(pred_lng), pred_x, pred_y


def run_random_agent():
  batch_size = 8

  ref360env = Refer360Batch(batch_size=batch_size, splits=['validation.seen'])

  ref360env.reset_epoch()
  n_epochs = 1
  max_step = 10
  repeat = (3, 5)
  threshold = 1.0 / max_step

  for epoch in range(n_epochs):
    im_batch, observations = ref360env.next()

    done_list = [False]*batch_size
    distances = [0]*batch_size
    history = {}

    for o in observations:
      history[o['id']] = {'lat_diffs': [],
                          'lng_diffs': [],
                          'start_lng': float(o['latitude']),
                          'start_lat': float(o['longitude']),
                          'pano': o['pano'],
                          'pred_lat': 0,
                          'pred_lng': 0}

    for _step in range(max_step):
      actions = []

      for ii, o in enumerate(observations):
        if done_list[ii]:
          continue

        sample = np.random.uniform(0, 1)
        pred_lat = 0.
        pred_lng = 0.
        argmax_x, armgax_y = 0, 0
        pred = np.zeros((FOV_SIZE, FOV_SIZE))
        if sample >= threshold or _step <= max_step / 2:
          navigate = torch.tensor(np.array(np.random.choice([1, 2, 3, 4])))
          lng_diff = __look__[navigate.item()][1]
          lat_diff = __look__[navigate.item()][0]

          history[o['id']]['lng_diffs'].append(lng_diff)
          history[o['id']]['lat_diffs'].append(lat_diff)
        else:
          navigate = torch.tensor(np.array(0))
          pred, pred_lat, pred_lng, argmax_x, armgax_y = predict_location(
              o, history)
          done_list[ii] = True
          lat_diff = 0
          lng_diff = 0

          distances[ii] = (pred_lat - o['latitude'])**2 + \
              (pred_lng - o['longitude'])**2

        actions.append((navigate.cuda(),
                        torch.tensor([argmax_x, armgax_y]).cuda(),
                        torch.tensor(pred).float().cuda()))

      im_batch, _, _, observations = ref360env.step(actions)
      if all(done_list):
        break

    for ii, o in enumerate(observations):
      if done_list[ii]:
        continue
      _, pred_lat, pred_lng, _, _ = predict_location(o, history)
      distances[ii] = (pred_lat - o['latitude'])**2 + \
          (pred_lng - o['longitude'])**2


if __name__ == '__main__':
  np.random.seed(0)
  run_random_agent()
