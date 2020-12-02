def get_localizer(args, n_vocab):
  '''Return a localizer module.
  '''
  from models.td_models import Concat
  from models.td_models import ConcatConv
  from models.td_models import RNN2Conv
  from models.td_models import LingUNet
  from models.visualbert import VisualBert
  from models.lxmert import LXMERTLocalizer

  args.use_raw_image = False
  rnn_args = {}
  rnn_args['input_size'] = n_vocab
  rnn_args['embed_size'] = args.n_emb
  rnn_args['rnn_hidden_size'] = int(args.n_hid /
                                    2) if args.bidirectional else args.n_hid
  rnn_args['num_rnn_layers'] = args.n_layers
  rnn_args['embed_dropout'] = args.dropout
  rnn_args['bidirectional'] = args.bidirectional
  rnn_args['reduce'] = 'last' if not args.bidirectional else 'mean'

  cnn_args = {}
  out_layer_args = {'linear_hidden_size': args.n_hid,
                    'num_hidden_layers': args.n_layers}
  image_channels = args.n_img_channels

  if args.model == 'concat':
    model = Concat(rnn_args, out_layer_args,
                   image_channels=image_channels)

  elif args.model == 'concat_conv':
    cnn_args = {'kernel_size': 5, 'padding': 2,
                'num_conv_layers': args.n_layers, 'conv_dropout': args.dropout}
    model = ConcatConv(rnn_args, cnn_args, out_layer_args,
                       image_channels=image_channels)

  elif args.model == 'rnn2conv':
    assert args.n_layers is not None
    cnn_args = {'kernel_size': 5, 'padding': 2,
                'conv_dropout': args.dropout}
    model = RNN2Conv(rnn_args, cnn_args, out_layer_args,
                     args.n_layers,
                     image_channels=image_channels)

  elif args.model == 'lingunet':
    assert args.n_layers is not None
    cnn_args = {'kernel_size': 5, 'padding': 2,
                'deconv_dropout': args.dropout}
    model = LingUNet(rnn_args, cnn_args, out_layer_args,
                     m=args.n_layers,
                     image_channels=image_channels)
  elif args.model == 'visualbert':
    args.use_masks = True
    model = VisualBert(args, n_vocab)
  elif args.model == 'lxmert':
    args.use_raw_image = True
    model = LXMERTLocalizer(args)
  else:
    raise NotImplementedError('Model {} is not implemented'.format(args.model))
  print('using {} localizer'.format(args.model))
  return model.cuda()
