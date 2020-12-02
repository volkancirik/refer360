"""
RL agent playing the game
"""
import paths
from env import Refer360Batch
from src.utils import get_model
from arguments import get_train_rl
import numpy as np
from collections import namedtuple
import json
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tensorboardX import SummaryWriter
from tqdm import tqdm
from collections import defaultdict

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
eps = np.finfo(np.float32).eps.item()
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')
#device = torch.device('cuda')


def select_action(model, state, done_list,
                  random_agent=False,
                  greedy=False,
                  rnn_hidden=None,
                  save_actions=True):
  """Sample actions using the model for a given state.
  """

  model_out = model(state, rnn_hidden=rnn_hidden)
  action_logits, state_values, loc_pred = model_out[
      'action_logits'], model_out['state_values'], model_out['loc_pred']

  if random_agent:
    sampling_logits = torch.ones(*action_logits.size()).cuda()
  else:
    sampling_logits = action_logits

  sampling_prob = F.softmax(sampling_logits, dim=-1)
  m = Categorical(sampling_prob)
  if greedy:
    _, navigate = torch.max(sampling_prob, 1)
  else:
    navigate = m.sample()

  maxes = loc_pred.view(loc_pred.size(0), -1).argmax(1).view(-1, 1)
  indices = torch.cat((maxes // loc_pred.size(2), maxes %
                       loc_pred.size(2)), dim=1)

  actions = []
  log_probs = m.log_prob(navigate)
  for bb in range(navigate.size(0)):
    actions.append((navigate[bb], indices[bb], loc_pred[bb]))
    if bb not in done_list and save_actions:
      model.saved_actions[bb].append(
          SavedAction(log_probs[bb], state_values[bb]))

  out = {'action_prob': sampling_prob,
         'action_logits': sampling_logits,
         'state_values': state_values,
         'loc_pred': loc_pred,
         'actions': actions}
  return out, model_out['rnn_hidden']


def calculate_policy_loss(saved_actions, rewards, gamma=0.99):
  """Policy loss for one instance.
  """
  R = 0
  returns = []
  for r in rewards[:: -1]:
    R = r + gamma * R
    returns.insert(0, R)
  returns = torch.tensor(returns)
  if len(returns) == 1:
    std = 1
  else:
    std = returns.std()
  #returns = (returns - returns.mean()) / (std + eps)

  policy_loss = 0.0
  for (log_prob, value), R in zip(saved_actions, returns):
    policy_loss = policy_loss + (-log_prob * R)
  return policy_loss, torch.tensor(0.0).cuda()


def calculate_actor_critic(saved_actions, rewards, gamma=0.99):
  """Advantage-Actor-Critic loss for one instance.
  """

  R = 0
  returns = []
  for r in rewards[:: -1]:
    R = r + gamma * R
    returns.insert(0, R)

  returns = torch.tensor(returns)

  if len(returns) == 1:
    std = 1
  else:
    std = returns.std()
  #returns = (returns - returns.mean()) / (std + eps)

  policy_loss = 0.0
  value_loss = 0.0

  for (log_prob, value), R in zip(saved_actions, returns):
    advantage = R - value.item()

    policy_loss = policy_loss + (-log_prob * advantage)
    value_loss = value_loss + \
        (F.smooth_l1_loss(value, torch.tensor([R]).cuda()))
  return policy_loss, value_loss


def finish_episode(model, optimizer, clip, update=False):
  """Finish the episode and backprop if update is true
  """

  policy_loss = 0.0
  value_loss = 0.0
  for bb in range(model.batch_size):

    rewards = model.rewards[bb]
    saved_actions = model.saved_actions[bb]
    #pl, vl = calculate_policy_loss(saved_actions, rewards)
    pl, vl = calculate_actor_critic(saved_actions, rewards)
    policy_loss += pl
    value_loss += vl

  loss = policy_loss + value_loss
  if update:
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()

  del model.rewards
  del model.saved_actions
  model.rewards = [[] for bb in range(model.batch_size)]
  model.saved_actions = [[] for bb in range(model.batch_size)]

  return {'loss': loss.item(),
          'policy_loss': policy_loss.item(),
          'value_loss': value_loss.item()}


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
  batch_size = args.batch_size
  verbose = args.verbose

  ref360env.reset_epoch()
  im_batch, observations = ref360env.next()

  n_updates = int(len(ref360env) / args.batch_size)
  if verbose:
    pbar = tqdm(range(n_updates))
  else:
    pbar = range(n_updates)

  for bid in pbar:
    done_list = [False]*batch_size

    batch_metrics = defaultdict(list)
    for metric in ['reward']:
      batch_metrics[metric] = [list() for kk in range(batch_size)]
    done_set = set()

    rnn_hidden = torch.zeros(args.n_layers,
                             batch_size, args.n_hid).cuda()

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

    for _step in range(max_step):
      state = {'im_batch': im_batch, 'observations': observations}
      out, rnn_hidden = select_action(model, state, done_set,
                                      rnn_hidden=rnn_hidden,
                                      greedy=args.greedy,
                                      random_agent=args.random_agent)

      actions = out['actions']
      im_batch, rewards, done_list, observations = ref360env.step(
          actions, life_penalty=-(_step+1))

      for kk, o in enumerate(observations):
        if kk in done_set or update:
          continue

        #actions.append((navigate[bb], indices[bb], pred[bb]))
        act, prediction, _ = actions[kk]
        act = act.item()
        if act == 0:
          lng_diff = 0
          lat_diff = 0
          pixel_map = o['pixel_map']

          pred_x, pred_y = prediction.cpu().numpy().tolist()
          pred_lat = pixel_map[pred_y][pred_x][0]
          pred_lng = pixel_map[pred_y][pred_x][1]
          history[o['id']]['pred_x'] = int(pred_x)
          history[o['id']]['pred_y'] = int(pred_y)

          history[o['id']]['pred_lat'] = float(pred_lat)
          history[o['id']]['pred_lng'] = float(pred_lng)

        else:
          lat_diff = ref360env.env.__look__[act][0]
          lng_diff = ref360env.env.__look__[act][1]

        history[o['id']]['lng_diffs'].append(lng_diff)
        history[o['id']]['lat_diffs'].append(lat_diff)
      for kk, (d, r) in enumerate(zip(done_list, rewards)):
        if kk not in done_set:
          model.rewards[kk].append(r)
          batch_metrics['reward'][kk].append(r.item())
          if d:
            done_set.add(kk)

    for kk in range(batch_size):
      if kk not in done_set:
        # batch_metrics['reward'][kk] = -10000.0  # ref360env.UNFINISHED_REWARD
        done_list[kk] = True

    epoch_mean = defaultdict()
    for metric in batch_metrics.keys():
      mean_metric = np.mean([sum(batch_metrics[metric][kk])
                             for kk in range(batch_size)])
      epoch_metrics[metric] += [mean_metric]
      epoch_mean[metric] = np.mean(epoch_metrics[metric])

    metrics = ref360env.eval_batch()
    for metric in metrics:
      epoch_metrics[metric] += [metrics[metric]]
      epoch_mean[metric] = np.mean(epoch_metrics[metric])

    losses = finish_episode(
        model, optimizer, args.clip, update=update)
    im_batch, observations = ref360env.next()

    for loss_type in losses:
      epoch_losses[loss_type].append(losses[loss_type])

    epoch_mean_loss = np.mean(epoch_losses['loss'])
    log_loss = 'E-Loss {:3.2f} B-Loss {:3.2f}'.format(
        epoch_mean_loss, losses['loss'])

    logs = dict()
    log_list = [log_loss]
    for metric in sorted(epoch_mean):
      logs[metric] = '{} E: {:3.3f} B: {:3.3f}'.format(metric,
                                                       epoch_mean[metric],
                                                       epoch_metrics[metric][-1])
      log_list += [logs[metric]]
    log_string = '|'.join(log_list)

    loss_group = {loss_type: np.mean(
        epoch_losses[loss_type]) for loss_type in epoch_losses}

    log_split_batch = log_split + ' B{:4d}|{:4d}'.format(bid+1, n_updates)
    log_final = ' '.join([log_split_batch, log_string])
    if verbose:
      pbar.set_description(log_final)
    else:
      print(log_final)
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
  parser = get_train_rl()
  args = parser.parse_args()

  writer = SummaryWriter(os.path.join(args.exp_dir, args.prefix))

  trn_ref360env = Refer360Batch(batch_size=args.batch_size,
                                splits=['train'],
                                seed=args.seed,
                                degrees=args.degrees,
                                use_look_ahead=args.use_look_ahead,
                                oracle_mode=args.oracle_mode,
                                images=args.trn_images,
                                prepare_vocab=True)
  val_ref360env = Refer360Batch(batch_size=args.batch_size,
                                splits=['validation.unseen'],
                                seed=args.seed,
                                degrees=args.degrees,
                                use_look_ahead=args.use_look_ahead,
                                oracle_mode=args.oracle_mode,
                                images=args.val_images)

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
