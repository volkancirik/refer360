"""
TF agent playing the game
"""
import paths

from get_model import get_model
from env import Refer360Batch

from arguments import get_train_rl
import numpy as np
from collections import namedtuple
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter
from tqdm import tqdm
from collections import defaultdict
from model_utils import smoothed_gaussian

from train_rl import select_action
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
eps = np.finfo(np.float32).eps.item()


def eval_epoch(ref360env, model, optimizer, args,
               update=False,
               log_split='',
               split_name='',
               n_iter=0,
               epoch=0,
               writer=None,
               sample_path='',
               debug=False):
  """Pass one epoch over the data split.
  """

  if update:
    model.train()
  else:
    model.eval()
  epoch_losses = defaultdict(list)
  epoch_metrics = defaultdict(list)

  max_step = args.max_step
  batch_size = args.batch_size
  verbose = args.verbose

  ref360env.reset_epoch()
  im_batch, observations = ref360env.next()

  n_updates = int(len(ref360env) / args.batch_size) if not debug else debug
  if verbose:
    pbar = tqdm(range(n_updates))
  else:
    pbar = range(n_updates)

  CELoss = nn.CrossEntropyLoss(reduction='none')

  for bid in pbar:
    log_split_batch = log_split + ' B{:2d}|{:3d}'.format(bid, n_updates)
    done_list = [False]*batch_size

    batch_metrics = defaultdict(list)
    for metric in ['reward']:
      batch_metrics[metric] = [list() for kk in range(batch_size)]
    done_set = set()
    rnn_hidden = torch.zeros(args.n_layers,
                             batch_size, args.n_hid).cuda()

    history = {}
    for o in observations:
      history[o['id']] = {'id': o['id'],
                          'lat_diffs': [],
                          'lng_diffs': [],
                          'start_lng': float(o['longitude']),
                          'start_lat': float(o['latitude']),
                          'pano': o['pano'],
                          'pred_lat': 0,
                          'pred_lng': 0,
                          'pred_x': 0,
                          'pred_y': 0,
                          'gt_lat': float(o['gt_lat']),
                          'gt_lng': float(o['gt_lng']),
                          'refexps': o['refexps']}

    losses = defaultdict(list)
    if not update:
      losses['loss'] = [0]
      losses['pred'] = [0]
      losses['action'] = [0]

    debug_act = np.ones((batch_size, max_step)) * -1
    for _step in range(max_step):
      state = {'im_batch': im_batch, 'observations': observations}
      out, _ = select_action(model, state, done_set,
                             rnn_hidden=rnn_hidden,
                             greedy=args.greedy,
                             random_agent=args.random_agent,
                             save_actions=False)

      if update:
        gt_tuples = [o['gt_tuple'] for o in state['observations']]
        gt_coors = [o['gt_coor'] for o in state['observations']]
        gt_actions = torch.tensor([o['gt_tuple'][0].item()
                                   for o in state['observations']]).cuda()

        loss_action = CELoss(out['action_logits'],
                             gt_actions)
        loss_mask = []
        loss_pred = 0
        for bb in range(batch_size):
          if gt_actions[bb] == 0 and bb not in done_list:

            loss_pred += smoothed_gaussian(
                out['loc_pred'], int(
                    gt_coors[bb][1]/4), int(gt_coors[bb][0]/4),
                height=100,
                width=100)
          if bb not in done_set:
            loss_mask += [1]
          else:
            loss_mask += [0]
        loss_mask = torch.tensor(loss_mask).cuda()
        loss_action_total = torch.sum(loss_action * loss_mask)
        loss = loss_action_total + loss_pred
        losses['action'].append(loss_action_total.item())
        if loss_pred != 0:
          losses['pred'].append(loss_pred.item())
        losses['loss'].append(loss.item())
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        actions = gt_tuples
      else:
        actions = out['actions']

        for bbb, a in enumerate(out['actions']):
          debug_act[bbb, _step] = a[0]

      im_batch, rewards, done_list, observations = ref360env.step(
          actions, life_penalty=-(_step+1))

      for kk, o in enumerate(observations):
        if kk in done_set or update:
          continue

        act, prediction, _ = actions[kk]
        act = act.item()
        if act == 0:
          lng_diff = 0
          lat_diff = 0
          pixel_map = o['pixel_map']

          pred_x, pred_y = prediction.cpu().numpy().tolist()
          pred_lng = pixel_map[pred_y][pred_x][0]
          pred_lat = pixel_map[pred_y][pred_x][1]
          history[o['id']]['pred_x'] = int(pred_x)
          history[o['id']]['pred_y'] = int(pred_y)

          history[o['id']]['pred_lat'] = float(pred_lat)
          history[o['id']]['pred_lng'] = float(pred_lng)

        else:
          lng_diff = ref360env.env.__look__[act][0]
          lat_diff = ref360env.env.__look__[act][1]

        history[o['id']]['lng_diffs'].append(lng_diff)
        history[o['id']]['lat_diffs'].append(lat_diff)

      for kk, (d, r) in enumerate(zip(done_list, rewards)):
        if kk not in done_set:
          model.rewards[kk].append(r)
          batch_metrics['reward'][kk].append(r.item())
          if d:
            done_set.add(kk)

    epoch_mean = defaultdict()
    metrics = ref360env.eval_batch()
    for metric in metrics:
      epoch_metrics[metric] += [metrics[metric]]
      epoch_mean[metric] = np.mean(epoch_metrics[metric])
    for metric in batch_metrics.keys():
      mean_metric = np.mean([sum(batch_metrics[metric][kk])
                             for kk in range(batch_size)])

      epoch_metrics[metric] += [mean_metric]
      epoch_mean[metric] = np.mean(epoch_metrics[metric])

    im_batch, observations = ref360env.next()

    for loss_type in losses:
      epoch_losses[loss_type].append(np.mean(losses[loss_type]))

    epoch_mean_loss = np.mean(epoch_losses['loss'])
    log_loss = 'E-Mean Loss {:3.2f} B-Loss {:3.2f}'.format(
        epoch_mean_loss, np.mean(losses['loss']))

    logs = dict()
    log_list = [log_loss]
    for metric in epoch_mean:
      logs[metric] = '{} E: {:2.2f} B: {:2.2f}'.format(metric,
                                                       epoch_mean[metric],
                                                       epoch_metrics[metric][-1])
      log_list += [logs[metric]]
    log_string = ' | '.join(log_list)

    loss_group = {loss_type: np.mean(
        epoch_losses[loss_type]) for loss_type in epoch_losses}

    if verbose:
      pbar.set_description(' '.join([log_split_batch, log_string]))
    else:
      print(' '.join([log_split_batch, log_string]))
    # add stats to tensorboard
    for loss_type in losses:
      writer.add_scalar('{}_{}'.format(
          split_name, loss_type), np.mean(losses[loss_type]), n_iter)

    if not update:
      for index in history.keys():
        h = history[index]
        with open(os.path.join(sample_path, '{}.E{}.json'.format(index, epoch)), 'w') as outfile:
          json.dump(h, outfile)

    n_iter += 1

  # add stats to tensorboard
  for metric in epoch_mean.keys():
    writer.add_scalar('data/{}_mean_{}'.format(split_name,
                                               metric), epoch_mean[metric], epoch)
  for loss_type in loss_group:
    writer.add_scalar(
        'data/{}_{}'.format(split_name, loss_type),
        loss_group[loss_type],
        epoch)
  epoch_mean['loss'] = epoch_mean_loss
  if verbose:
    pbar.close()
  return epoch_mean, n_iter


def main():
  parser = get_train_rl()
  args = parser.parse_args()

  writer = SummaryWriter(os.path.join(args.exp_dir, args.prefix))

  trn_split = ['validation.unseen'] if args.debug else ['train']

  trn_ref360env = Refer360Batch(batch_size=args.batch_size,
                                splits=trn_split,
                                seed=args.seed,
                                degrees=args.degrees,
                                oracle_mode=args.oracle_mode,
                                use_gt_action=True,
                                prepare_vocab=True)

  val_ref360env = Refer360Batch(batch_size=args.batch_size,
                                splits=['validation.unseen'],
                                seed=args.seed,
                                degrees=args.degrees,
                                oracle_mode=args.oracle_mode,
                                use_gt_action=True)

  model = get_model(args, trn_ref360env.vocab, n_actions=5)
  optimizer = optim.Adam(model.parameters(),
                         lr=args.lr,
                         weight_decay=args.weight_decay)

  best_val = float('-inf')
  sample_path = os.path.join(args.exp_dir, args.prefix, 'samples/')
  os.makedirs(sample_path, exist_ok=True)

  for epoch in range(args.epoch):
    tot_trn_iter = 0
    tot_val_iter = 0

    trn_log = 'trn E{:2d}'.format(epoch)
    _, n_iter = eval_epoch(trn_ref360env, model, optimizer, args,
                           update=True,
                           log_split=trn_log,
                           split_name='trn',
                           n_iter=tot_trn_iter,
                           epoch=epoch,
                           writer=writer,
                           sample_path='',
                           debug=args.debug)
    tot_trn_iter = n_iter

    if epoch % args.val_freq == 0:
      val_log = 'val E{:2d}'.format(epoch)
      val_metrics, n_iter = eval_epoch(val_ref360env, model, optimizer, args,
                                       update=False,
                                       log_split=val_log,
                                       split_name='val',
                                       n_iter=tot_val_iter,
                                       epoch=epoch,
                                       writer=writer,
                                       sample_path=sample_path,
                                       debug=args.debug)
      tot_val_iter = n_iter
      if val_metrics[args.metric] > best_val:
        print('\n UPDATING MODELS! \n')
        best_val = val_metrics[args.metric]
        torch.save(model, os.path.join(
            args.exp_dir, args.prefix + '/model.pt'))


if __name__ == '__main__':
  main()
