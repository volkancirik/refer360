def get_model(args, vocab, n_actions=5):
  '''Returns a Finder model.
  '''
  from models.finder import Finder
  from models.finder_look_ahead import FinderLookAhead
  if args.model == 'visualbert':
    args.use_masks = True
    args.cnn_layer = 2
    args.n_emb = 512

  if args.use_look_ahead:
    finder = FinderLookAhead
  else:
    finder = Finder

  return finder(args, vocab, n_actions).cuda()
