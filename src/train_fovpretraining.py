"""
Pre-training VisualBert by predicting nearby objects.
"""
import paths
from arguments import get_train_fovpretraining

import numpy as np
import os
import json

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter
from tqdm import tqdm

from torch.utils.data import DataLoader
from fov_pretraining import FoVTask1
from fov_pretraining import FoVTask2
from fov_pretraining import FoVTask3
from fov_pretraining import FoVTask4

from models import Hallucinator
from models import LXMERTHallucinator

from utils import DIRECTIONS
from utils import get_object_dictionaries
from model_utils import get_confusion_matrix_image
from model_utils import F1_Binary_Loss
from model_utils import compute_precision_with_logits
from model_utils import weights_init_uniform_rule
from collections import Counter
from pprint import pprint


def get_fovpretraining_model(args, n_objects, n_directions,
                             use_queries=False):
  '''Returns a model.
  '''

  task = args.task
  use_queries = args.task in set(['task2', 'task3', 'task4'])
  if task == 'task1':
    n_actions = n_objects * n_directions
  elif task == 'task2':
    n_actions = n_directions
  elif task == 'task3' or task == 'task4':
    n_actions = 1
  else:
    raise NotImplementedError()

  if args.model == 'lxmert':
    return LXMERTHallucinator(args, n_actions).cuda()
  return Hallucinator(args.batch_size,
                      hidden_size=args.n_hid,
                      n_actions=n_actions,
                      n_objects=n_objects,
                      use_queries=use_queries,
                      fov_emb_mode=args.fov_emb_mode).cuda()


def eval_epoch(data_iterator, model, optimizer, args,
               update=False,
               log_split='',
               split_name='',
               n_iter=0,
               epoch=0,
               writer=None,
               sample_path='',
               debug=0,
               verbose=False,
               obj_classes=[],
               weights=[],
               clip=1.0,
               most_freq=0,
               log_path=''):
  """Pass one epoch over the data split.
  """

  if update:
    model.train()
  else:
    model.eval()

  task = args.task
  verbose = args.verbose

  num_classes = model.n_actions if model.n_actions > 1 else 2
  confusion = torch.zeros(num_classes, num_classes).float()

  if task == 'task1':
    matrix_labels = []
    for dir in DIRECTIONS[args.direction]:
      for obj in obj_classes:
        matrix_labels += [dir + '_' + obj]
  elif task == 'task2':
    matrix_labels = DIRECTIONS[args.direction]
  elif task in set(['task3', 'task4']):
    matrix_labels = ['not present', 'present']
  else:
    raise NotImplementedError()

  n_updates = len(data_iterator) if not debug else debug

  if verbose:
    pbar = tqdm(data_iterator)
  else:
    pbar = data_iterator

  total_loss = []
  total_f1 = []
  total_acc = []

  zeros_acc = []
  ones_acc = []
  random_acc = []
  mostfreq_acc = []

  f1_binary_loss = F1_Binary_Loss().cuda()

  cce_loss = nn.CrossEntropyLoss(weight=torch.tensor(weights)).cuda()

  for bid, (batch_lat,
            batch_lng,
            batch_images,
            batch_boxes,
            batch_feats,
            batch_queries,
            batch_query_strings,
            batch_targets,
            batch_most_freq) in enumerate(pbar):
    if debug and bid == n_updates:
      break
    out = model({
        'latitude': batch_lat,
        'longitude': batch_lng,
        'im_batch': batch_images,
        'obj_boxes': batch_boxes,
        'obj_feats': batch_feats,
        'queries': batch_queries,
        'query_strings': batch_query_strings})
    preds = out['action_logits']

    if task in set(['task1', 'task3', 'task4']):

      binary_preds = (preds > 0.5).float()
      binary_preds.requires_grad = True

      if task == 'task1':
        w = 10000.0
        weight_rebal = torch.ones_like(
            batch_targets) / w + (1.0 - 1.0 / w) * batch_targets

        loss_fn = nn.BCELoss(weight=weight_rebal)
      else:
        loss_fn = torch.nn.functional.binary_cross_entropy_with_logits

      loss = loss_fn(preds, batch_targets)

      f1_score = f1_binary_loss(binary_preds, batch_targets)
      acc = ((preds > 0.5).int() == batch_targets).float().mean()

      zero_acc = (torch.zeros_like(batch_targets)
                  == batch_targets).float().mean()

      one_acc = (torch.ones_like(batch_targets)
                 == batch_targets).float().mean()
      r_acc = (torch.empty(batch_targets.size()).random_(
          2).cuda() == batch_targets).float().mean()

      total_f1.append(f1_score.item())
      total_acc.append(acc.item())
      zeros_acc.append(zero_acc.item())
      ones_acc.append(one_acc.item())
      random_acc.append(r_acc.item())

      binary_preds = (preds > 0.5).long()
      binary_preds = Counter(['{}'.format(bp.item()) for bp in binary_preds])
      binary_preds = json.dumps(binary_preds)

      log_string = 'f1: {:3.3f} mean-acc: {:3.3f} 0-acc: {:3.3f} 1-acc: {:3.3f} r-acc: {:3.3f} preds : {:10s}'.format(
          np.mean(total_f1),
          np.mean(total_acc),
          np.mean(zeros_acc),
          np.mean(ones_acc),
          np.mean(random_acc),
          binary_preds)

      if task == 'task1':
        pred_indices = [[] for bb in range(preds.size(0))]
        for pair in (preds > 0.5).nonzero(as_tuple=False):
          pred_indices[pair[0]].append(pair[1].item())
      elif task in set(['task3', 'task4']):
        pred_indices = (preds > 0.5).long()
    elif task == 'task2':

      loss = cce_loss(preds, batch_targets.squeeze(1).long())

      acc = compute_precision_with_logits(
          preds, batch_targets.long())
      total_acc.append(acc.item())
      r_acc = (torch.empty(batch_targets.size()).random_(
          num_classes).cuda() == batch_targets).float().mean()
      random_acc.append(r_acc.item())

      mfreq_acc = (batch_most_freq == batch_targets).float().mean()
      mostfreq_acc.append(mfreq_acc.item())

      log_string = 'acc: {:3.3f} mfreq-acc: {:3.3f} r-acc: {:3.3f}'.format(
          np.mean(total_acc), np.mean(mostfreq_acc), np.mean(random_acc))

      _, pred_indices = torch.max(preds, 1)
    else:
      raise NotImplementedError()
    total_loss.append(loss.item())
    if update:
      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
      optimizer.step()
    log_string = 'mean-loss: {:3.3f} '.format(np.mean(total_loss)) + log_string
    log_split_batch = log_split + ' B{:4d}|{:4d}'.format(bid+1, n_updates)
    log_final = ' '.join([log_split_batch, log_string])
    if verbose:
      pbar.set_description(log_final)
    else:
      print(log_final)

    if batch_targets.size(1) == 1:
      targets = [[batch_targets[bb].long().item()]
                 for bb in range(preds.size(0))]
    else:
      targets = [[] for bb in range(preds.size(0))]
      for pair in batch_targets.nonzero(as_tuple=False):
        targets[pair[0]].append(pair[1].item())

    for bb in range(preds.size(0)):
      for t in targets[bb]:
        confusion[t, pred_indices[bb]] += 1

    total_score = total_f1 if task in set(['task1,task3,task4']) else total_acc
    writer.add_scalar('{}_{}'.format(
        split_name, 'batch_score'), total_score[-1], n_iter)
    writer.add_scalar('{}_{}'.format(
        split_name, 'batch_loss'), total_loss[-1], n_iter)

    n_iter += 1
    if (n_iter+1) % 100 == 0 and update:
      model_name = os.path.join(
          args.exp_dir, args.prefix + '/model.{}_{}.pt'.format(epoch, n_iter))
      print('\n saving model', model_name)
      torch.save(model, model_name)

  if verbose:
    pbar.close()

  writer.add_scalar('{}_{}'.format(
      split_name, 'epoch_score'), np.mean(total_score), epoch)
  writer.add_scalar('{}_{}'.format(
      split_name, 'epoch_loss'), np.mean(total_loss), epoch)

  img_conf = get_confusion_matrix_image(
      matrix_labels, confusion / confusion.sum(), '')

  writer.add_image('Confusion Matrix', img_conf, epoch)
  with open(log_path, "a") as log_file:
    log_file.write(log_final+'\n')
  return {'loss': np.mean(total_loss),
          'accuracy': np.mean(total_score)}, n_iter


def main():
  parser = get_train_fovpretraining()
  args = parser.parse_args()

  writer = SummaryWriter(os.path.join(args.exp_dir, args.prefix))

  if args.task == 'task1':
    Task = FoVTask1
  elif args.task == 'task2':
    Task = FoVTask2
  elif args.task == 'task3':
    Task = FoVTask3
  elif args.task == 'task4':
    Task = FoVTask4
  else:
    raise NotImplementedError('Task {} not implemented'.format(args.task))

  training_split = ['validation.seen'] if args.debug > 0 else [
      'train']
  validation_unseen_split = training_split if args.debug > 0 else [
      'validation.unseen']
  validation_seen_split = training_split if args.debug > 0 else [
      'validation.seen']

  trn_dataset = Task(training_split, args.direction,
                     data_root=args.data_root,
                     obj_dict_file=args.obj_dict_file,
                     use_objects=args.model in set(['lxmert']),
                     ignore_list=args.ignore_list)
  val_seen_dataset = Task(validation_seen_split, args.direction,
                          data_root=args.data_root,
                          obj_dict_file=args.obj_dict_file,
                          use_objects=args.model in set(['lxmert']),
                          ignore_list=args.ignore_list)
  val_unseen_dataset = Task(validation_unseen_split, args.direction,
                            data_root=args.data_root,
                            obj_dict_file=args.obj_dict_file,
                            use_objects=args.model in set(['lxmert']),
                            ignore_list=args.ignore_list)

  weights = [0] * len(DIRECTIONS[args.direction])
  for o in trn_dataset.labels.keys():
    for jj, dir in enumerate(DIRECTIONS[args.direction]):
      weights[jj] += trn_dataset.labels[o][dir]
  weights = [sum(weights)/v for v in weights]
  print('loss weighting:', weights)

  trn_iterator = DataLoader(
      dataset=trn_dataset,
      batch_size=args.batch_size,
      shuffle=args.debug == 0
  )

  val_seen_iterator = DataLoader(
      dataset=val_seen_dataset,
      batch_size=args.batch_size,
      shuffle=args.debug == 0
  )
  val_unseen_iterator = DataLoader(
      dataset=val_unseen_dataset,
      batch_size=args.batch_size,
      shuffle=args.debug == 0
  )

  vg2idx, idx2vg, obj_classes = get_object_dictionaries(args.obj_dict_file)

  model = get_fovpretraining_model(
      args, len(vg2idx), len(DIRECTIONS[args.direction]))
  model.apply(weights_init_uniform_rule)

  optimizer = optim.Adam(model.parameters(), lr=args.lr,
                         weight_decay=args.weight_decay)

  best_val = float('-inf')
  sample_path = os.path.join(args.exp_dir, args.prefix, 'samples/')
  os.makedirs(sample_path, exist_ok=True)

  tot_trn_iter = 0
  tot_val_seen_iter = 0
  tot_val_unseen_iter = 0

  log_path = os.path.join(
      args.exp_dir, args.prefix + '/training.log')

  print(json.dumps(vars(args), indent=2))
  for epoch in range(args.epoch):

    trn_log = 'trn E{:2d}'.format(epoch)
    _, n_iter = eval_epoch(trn_iterator, model, optimizer, args,
                           update=True,
                           log_split=trn_log,
                           split_name='trn',
                           n_iter=tot_trn_iter,
                           epoch=epoch,
                           writer=writer,
                           sample_path='',
                           obj_classes=obj_classes,
                           weights=weights,
                           clip=args.clip,
                           debug=args.debug,
                           log_path=log_path)
    tot_trn_iter = n_iter

    if epoch % args.val_freq == 0:
      val_log = 'val E{:2d}'.format(epoch)
      val_seen_metrics, n_iter = eval_epoch(val_seen_iterator, model, optimizer, args,
                                            update=False,
                                            log_split=val_log,
                                            split_name='v_seen',
                                            n_iter=tot_val_seen_iter,
                                            epoch=epoch,
                                            writer=writer,
                                            sample_path=sample_path,
                                            obj_classes=obj_classes,
                                            weights=weights,
                                            debug=args.debug,
                                            log_path=log_path)
      tot_val_seen_iter = n_iter
      val_unseen_metrics, n_iter = eval_epoch(val_unseen_iterator, model, optimizer, args,
                                              update=False,
                                              log_split=val_log,
                                              split_name='v_unseen',
                                              n_iter=tot_val_unseen_iter,
                                              epoch=epoch,
                                              writer=writer,
                                              sample_path=sample_path,
                                              obj_classes=obj_classes,
                                              weights=weights,
                                              debug=args.debug,
                                              log_path=log_path)
      tot_val_unseen_iter = n_iter
      if val_seen_metrics[args.metric] > best_val:
        model_name = os.path.join(
            args.exp_dir, args.prefix + '/model.best.pt')
        print('\n UPDATING BEST MODEL! {}\n'.format(model_name))
        best_val = val_seen_metrics[args.metric]
        torch.save(model, model_name)


if __name__ == '__main__':
  main()
