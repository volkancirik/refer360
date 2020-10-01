"""Utils for io, tokenization etc.
"""
import json
import numpy as np

import torch
import torch.nn as nn
from collections import defaultdict
import math

import os
from tqdm import tqdm


class PositionalEncoding(nn.Module):
  "Implement the PE function."

  def __init__(self, d_model, dropout, max_len=5000):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)

    # Compute the positional encodings once in log space.
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                         -(math.log(10000.0) / d_model))

    size_sin = pe[:, 0::2].size()
    pe[:, 0::2] = torch.sin(position * div_term)[:size_sin[0], : size_sin[1]]
    size_cos = pe[:, 1::2].size()
    pe[:, 1::2] = torch.cos(position * div_term)[:size_cos[0], : size_cos[1]]
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)

  def forward(self, x):
    pos_info = torch.tensor(self.pe[:, : x.size(1)],
                            requires_grad=False)
    x = x + pos_info
    return self.dropout(x)


def rad2degree(lat, lng,
               adjust=False):
  '''Convert radians to degrees.
  '''

  adj_lat = 0
  adj_lng = 0
  if adjust:
    adj_lat = -0.015
    adj_lng = 0.1

  lng = np.degrees(lng + adj_lng)
  lat = np.degrees(lat - adj_lat)
  if lat > 180:
    lat = lat - 360
  return lat, lng


def get_graph_hops(nodes, actions, image_path,
                   full_w=4552,
                   full_h=2276,
                   fov_size=512):
  '''Generate ground-truth paths for graph-based navigation.
  '''

  all_fovs = []

  for ii, inst in enumerate(actions[-1]['action_list'].split('_')):
    kk = 0
    prev_lat, prev_lng = -1, -1
    fovs = []
    for jj, fov in enumerate(inst.split('|')[1:]):
      rlat, rlng = float(fov.split(',')[0]), float(fov.split(',')[1])
      if prev_lat == rlat and prev_lng == rlng:
        continue

      prev_lat, prev_lng = rlat, rlng
      lat, lng = rad2degree(rlat, rlng)
      fx = int(full_w * ((lat + 180)/360.0))
      fy = int(full_h - full_h *
               ((lng + 90)/180.0))

      fovs.append((lat, lng, '{}.{}'.format(ii, kk), fx, fy))
      kk += 1
    all_fovs.append(fovs)

  gt_hops = []

  for fovs in all_fovs:

    closest_distances = []
    closest_nodes = []
    for fov in fovs:
      closest_d = full_h + full_w

      _, _, _, fx, fy = fov[0], fov[1], fov[2], fov[3], fov[4]

      for jj, n in enumerate(nodes.keys()):
        ox, oy = nodes[n]['x'], nodes[n]['y']

        d = ((fx-ox)**2 + (fy-oy)**2)**0.5
        if d < closest_d:
          closest_d = d
          closest = jj
      closest_distances.append(closest_d)
      closest_nodes.append(closest)
    gt_hops.append(closest_nodes)

  fov_dict = dict()

  kk = 0
  for sid, fovs in enumerate(all_fovs):
    for hid, fov in enumerate(fovs):
      n = gt_hops[sid][hid]

      f = {
          'lat': fov[0],
          'lng': fov[1],
          'label': fov[2],
          'x': fov[3],
          'y': fov[4],
          'closest_node': n,
          'sentence_id': sid,
          'hop_id': hid,
      }
      fov_dict[kk] = f
      kk += 1

  new_nodes = dict()
  for jj, n in enumerate(nodes.keys()):
    node = nodes[n]
    new_nodes[n] = node
    new_nodes[n]['neighbors'] = sorted([n]+list(set(node['neighbors'])))
  return fov_dict, gt_hops, new_nodes


def get_moves(instance, gt_lat, gt_lng, n_sentences):
  '''Return ground-truth lat/lng.
  '''

  all_moves = []
  flag = False
  for action in instance['actions']:
    if len(action['action_list'].split('_')) != n_sentences:
      flag = True
      break
    gt_moves = []
    for moves in action['action_list'].split('_'):
      move = moves.split('|')[-1]

      latitude, longitude = rad2degree(
          float(move.split(',')[0]), float(move.split(',')[1]))

      gt_moves.append((latitude, longitude))

      dist = ((gt_moves[-1][0] - gt_lat)**2 +
              (gt_moves[-1][1] - gt_lng)**2) ** 0.5
      if dist < 1:
        continue
      else:
        all_moves.append(gt_moves)
  return flag, all_moves


def coord_gaussian(x, y, gt_x, gt_y, sigma):
  '''Return the gaussian-weight of a pixel based on distance.
  '''
  pixel_val = (1 / (2 * np.pi * sigma ** 2)) * \
      np.exp(-((x - gt_x) ** 2 + (y - gt_y) ** 2) / (2 * sigma ** 2))
  return pixel_val if pixel_val > 1e-5 else 0.0


def gaussian_target(gt_x, gt_y, sigma=3.0,
                    width=400, height=400):
  '''Based on https://github.com/lil-lab/touchdown/tree/master/sdr
  '''
  target = torch.tensor([[coord_gaussian(x, y, gt_x, gt_y, sigma)
                          for x in range(width)] for y in range(height)]).float().cuda()
  return target


def smoothed_gaussian(pred, gt_x, gt_y,
                      sigma=3.0,
                      height=400, width=400):
  '''KLDivLoss for pixel prediction.
  '''

  loss_func = nn.KLDivLoss(reduction='sum')

  target = gaussian_target(gt_x, gt_y,
                           sigma=sigma, width=width, height=height)
  loss = loss_func(pred.unsqueeze(0), target.unsqueeze(0))
  return loss


def load_datasets(splits, image_categories='all',
                  root='../data/dumps'):
  '''Loads a dataset dump.
  '''
  d, s = [], []
  for split_name in splits:
    fname = os.path.join(
        root, '{}.[{}].imdb.npy'.format(split_name, image_categories))
    print('loading split from {}'.format(fname))
    dump = np.load(fname, allow_pickle=True)[()]
    d += dump['data_list']
    s += dump['sentences']

  return d, s


def dump_datasets(splits, image_categories, output_file,
                  graph_root=''):
  '''Prepares and dumps dataset to an npy file.
  '''

  banned_turkers = set(['onboarding', 'vcirik'])
  data_list = []

  cat2loc = {'restaurant': 'indoor',
             'shop': 'indoor',
             'expo_showroom': 'indoor',
             'living_room': 'indoor',
             'bedroom': 'indoor',
             'street': 'outdoor',
             'plaza_courtyard': 'outdoor',
             }
  all_sentences = []

  image_set = set(cat2loc.keys())
  if image_categories != 'all':
    image_set = set(image_categories.split(','))
    for image in image_set:
      if image not in cat2loc:
        raise NotImplementedError(
            'Image Category {} is not in the dataset'.format(image))
  stats = defaultdict(int)
  for split in splits:
    data = []
    instances = json.load(open('../data/{}.json'.format(split), 'r'))

    pbar = tqdm(instances)
    for ii, instance in enumerate(pbar):
      if instance['turkerid'] not in banned_turkers:
        if 'ann_cat' in instance and instance['ann_cat'] == 'M' and split == 'train':
          continue
        datum = {}

        # FIX
        latitude, longitude = rad2degree(instance['longitude'], instance['latitude'],
                                         adjust=True)

        datum['annotationid'] = instance['annotationid']
        datum["gt_lng"] = longitude
        datum["gt_lat"] = latitude

        img_category = '_'.join(instance['imageurl'].split(
            '/')[-1].split('.')[0].split('_')[2:])
        pano_name = "_".join(instance['imageurl'].split(
            '/')[-1].split('.')[0].split('_')[:2]) + ".jpg"

        if img_category not in cat2loc or img_category not in image_set:
          continue
        img_loc = cat2loc[img_category]
        datum['pano'] = '../data/refer360images/{}/{}/{}'.format(
            img_loc, img_category, pano_name)
        datum['img_category'] = img_category
        stats[img_loc] += 1
        stats[img_category] += 1
        sentences = []
        sent_queue = []
        for refexp in instance['refexp'].replace('\n', '').split('|||')[1:]:
          if refexp[-1] not in ['.', '!', '?', '\'']:
            refexp += '.'
          sentences.append(refexp.split())
          sent_queue += [refexp]

        err, all_moves = get_moves(
            instance, latitude, longitude, len(sentences))

        if err or all_moves == [] or len(all_moves[0]) != len(sentences):
          continue

        pano = "_".join(instance['imageurl'].split(
            '/')[-1].split('.')[0].split('_')[:2])

        datum['gt_moves'] = all_moves
        datum['refexps'] = sentences
        all_sentences += sent_queue

        if graph_root != '':
          node_path = os.path.join(graph_root, '{}.npy'.format(pano))
          node_img = os.path.join(graph_root, '{}.jpg'.format(pano))
          fov_prefix = os.path.join(graph_root, '{}.fov'.format(pano))
          nodes = np.load(node_path, allow_pickle=True)[()]
          fovs, graph_hops, new_nodes = get_graph_hops(nodes,
                                                       instance['actions'],
                                                       datum['pano'])
          assert len(sentences) == len(graph_hops)
          datum['fovs'] = fovs
          datum['graph_hops'] = graph_hops
          datum['nodes'] = new_nodes
          datum['node_img'] = node_img
          datum['fov_prefix'] = fov_prefix

        data.append(datum)
    data_list.append(data)
    pbar.close()

  print('_'*20)
  for c in stats:
    print('{:<16} {:2.2f}'.format(c, stats[c]))
  print('_'*20)

  print('Dumping to {}'.format(output_file))
  np.save(open(output_file, 'wb'), {'data_list': data_list,
                                    'sentences': all_sentences})
#  return data_list, all_sentences


def vectorize_seq(sequences, vocab, emb_dim, cuda=False, permute=False):
  '''Generates one-hot vectors and masks for list of sequences.
  '''
  vectorized_seqs = [vocab.encode(sequence) for sequence in sequences]

  seq_lengths = torch.tensor(list(map(len, vectorized_seqs)))
  seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
  max_length = seq_tensor.size(1)

  emb_mask = torch.zeros(len(seq_lengths), max_length, emb_dim)
  for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
    seq_tensor[idx, :seqlen] = torch.tensor(seq)
    emb_mask[idx, :seqlen, :] = 1.

  idx = torch.arange(max_length).unsqueeze(0).expand(seq_tensor.size())
  len_expanded = seq_lengths.unsqueeze(1).expand(seq_tensor.size())
  mask = (idx >= len_expanded)

  if permute:
    seq_tensor = seq_tensor.permute(1, 0)
    emb_mask = emb_mask.permute(1, 0, 2)
  if cuda:
    return seq_tensor.cuda(), mask.cuda(), emb_mask.cuda(), seq_lengths.cuda()
  return seq_tensor, mask, emb_mask, seq_lengths


def add_overlay(src, trg, coor,
                maxs_x=400, maxs_y=400,
                maxt_x=60, maxt_y=60,
                ignore=[76, 112, 71]):
  '''Adds an overlay of waldo icon for oracle experiments.
  '''
  xs_min = max(0, coor[1] - int(maxt_x / 2))

  xs_max = min(maxs_x, coor[1] + int(maxt_x / 2))
  ys_min = max(0, coor[0] - int(maxt_y / 2))
  ys_max = min(maxs_y, coor[0] + int(maxt_y / 2))

  xt_min = max(0, int(maxt_x / 2) - coor[1])
  xt_max = maxt_x - min(coor[1] + int(maxt_x / 2) - maxs_x, 0)
  yt_min = max(0, int(maxt_y / 2) - coor[0])
  yt_max = maxt_y - max(coor[0] + int(maxt_y / 2) - maxs_y, 0)

  if ignore == []:
    src[ys_min:ys_max,
        xs_min:xs_max] = trg[yt_min:yt_max,
                             xt_min:xt_max]

  else:
    for xs, xt in zip(range(xs_min, xs_max), range(xt_min, xt_max)):
      for ys, yt in zip(range(ys_min, ys_max), range(yt_min, yt_max)):
        pixel = trg[yt, xt, :].tolist()
        if pixel != ignore:
          src[ys, xs, :] = trg[yt, xt, :]
  return src


from src.models.finder import Finder


def get_model(args, vocab, n_actions=5):
  '''Returns a Finder model.
  '''
  if args.model == 'visualbert':
    args.use_masks = True
    args.cnn_layer = 2
    args.n_emb = 512

  return Finder(args, vocab, n_actions).cuda()
