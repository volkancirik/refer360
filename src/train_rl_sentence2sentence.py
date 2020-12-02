"""
RL agent playing the game
"""
import paths
from get_model import get_model
from env import Refer360Batch

import numpy as np
from collections import namedtuple
import json
import os
import torch

import torch.optim as optim

from tensorboardX import SummaryWriter
from tqdm import tqdm
from collections import defaultdict

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
eps = np.finfo(np.float32).eps.item()

from train_rl import select_action
from train_rl import finish_episode
from arguments import get_train_rl_sentence2sentence


def eval_epoch(ref360env, model, optimizer, args,
               update=False,
               log_split='',
               split_name='',
               n_iter=0,
               epoch=0,
               writer=None,
               sample_path=''):
  """Pass one epoch over the data split.
  """

  epoch_losses = defaultdict(list)
  epoch_metrics = defaultdict(list)

  max_step = args.max_step
  max_sentence = args.max_sentence
  batch_size = args.batch_size
  verbose = args.verbose

  ref360env.reset_epoch()
  im_batch, observations = ref360env.next()

  n_updates = int(len(ref360env) / args.batch_size)
  if verbose:
    pbar = tqdm(range(n_updates))
  else:
    pbar = range(n_updates)

  for _ in pbar:

    batch_metrics = defaultdict(list)
    for metric in ['reward']:
      batch_metrics[metric] = [list() for kk in range(batch_size)]

    history = {}
    for o in observations:
      history[o['id']] = {'lat_diffs': [],
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

    for _sentence in range(max_sentence):
      done_set = set()
      done_list = [False]*batch_size
      rnn_hidden = torch.zeros(args.n_layers,
                               batch_size, args.n_hid).cuda()

      for _step in range(max_step):
        state = {'im_batch': im_batch, 'observations': observations}

        out, rnn_hidden = select_action(model, state, done_set,
                                        rnn_hidden=rnn_hidden,
                                        greedy=args.greedy,
                                        random_agent=args.random_agent)

        actions = out['actions']

        im_batch, rewards, done_list, observations = ref360env.step(
            actions, life_penalty=-(_step+1))

        for kk, (d, r) in enumerate(zip(done_list, rewards)):
          if kk not in done_set:
            model.rewards[kk].append(r)
            batch_metrics['reward'][kk].append(r.item())
            if d:
              done_set.add(kk)
        if all(done_list):
          break

      for kk in range(batch_size):
        if kk not in done_set:
          done_list[kk] = True
      losses = finish_episode(
          model, optimizer, args.clip, update=update)
      ref360env.nextSentence()
      im_batch, observations = ref360env.nextStartPoint()

      for loss_type in losses:
        epoch_losses[loss_type].append(losses[loss_type])

    im_batch, observations = ref360env.next()
    epoch_mean = defaultdict()
    for metric in batch_metrics.keys():
      mean_metric = np.mean([sum(batch_metrics[metric][kk])
                             for kk in range(batch_size)])
      epoch_metrics[metric] += [mean_metric]
      epoch_mean[metric] = np.mean(epoch_metrics[metric])

    epoch_mean_loss = np.mean(epoch_losses['loss'])
    log_loss = 'E-Mean Loss {:3.2f} B-Loss {:3.2f}'.format(
        epoch_mean_loss, losses['loss'])

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
      pbar.set_description(' '.join([log_split, log_string]))
    else:
      print(' '.join([log_split, log_string]))
    # add stats to tensorboard
    for loss_type in losses:
      writer.add_scalar('{}_{}'.format(
          split_name, loss_type), losses[loss_type], n_iter)

    if not update:
      for index in history.keys():
        h = history[index]
        with open(os.path.join(sample_path, '{}.E{}.json'.format(index, epoch)), 'w') as outfile:
          json.dump(h, outfile)

    n_iter += 1

  # add stats to tensorboard
  for metric in epoch_mean.keys():
    writer.add_scalar('data/{}_mean_{}'.format(split_name, metric),
                      epoch_mean[metric], epoch)
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
  parser = get_train_rl_sentence2sentence()
  args = parser.parse_args()
  writer = SummaryWriter(os.path.join(args.exp_dir, args.prefix))

  trn_ref360env = Refer360Batch(batch_size=args.batch_size,
                                splits=['train'],
                                seed=args.seed,
                                degrees=args.degrees,
                                use_sentences=True,
                                prepare_vocab=True)

  val_ref360env = Refer360Batch(batch_size=args.batch_size,
                                splits=['validation.unseen'],
                                seed=args.seed,
                                degrees=args.degrees,
                                use_sentences=True)
  model = get_model(args, trn_ref360env.vocab, n_actions=6)
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
                           sample_path='')
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
                                       sample_path=sample_path)
      tot_val_iter = n_iter
      if val_metrics[args.metric] > best_val:
        print('\n UPDATING MODELS! \n')
        best_val = val_metrics[args.metric]
        torch.save(model, os.path.join(
            args.exp_dir, args.prefix + '/model.pt'))


if __name__ == '__main__':
  main()
