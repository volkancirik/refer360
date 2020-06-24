from nltk.tokenize import TweetTokenizer
from collections import defaultdict
UNK_TOKEN = '__UNK__'
START_TOKEN = '__START__'
END_TOKEN = '__END__'
PAD_TOKEN = '__PAD__'
NO_TOKEN = '__NOSPEAK__'
SPECIALS = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN, NO_TOKEN]


def split_tokenize(text):
  """Splits tokens based on whitespace after adding whitespace around
  punctuation.
  """
  return (text.lower().replace('.', ' . ').replace('. . .', '...')
          .replace(',', ' , ').replace(';', ' ; ').replace(':', ' : ')
          .replace('!', ' ! ').replace('?', ' ? ')
          .split())


class Dictionary:
  '''Simple dictionary for conversion of strings to indexes, and vice versa.
'''

  def __init__(self, sentences, min_freq=0, split=False):
    self.i2tok = list()
    self.tok2i = dict()
    self.tok2cnt = defaultdict(int)
    self.split = split

    for tok in SPECIALS:
      self.tok2i[tok] = len(self.tok2i)
      self.i2tok.append(tok)
      self.tok2cnt[tok] = 100000000

    for line in sentences:
      toks = line.lower().split(' ')
      for tok in toks:
        self.tok2cnt[tok] += 1

    for tok in self.tok2cnt:
      if self.tok2cnt[tok] >= min_freq:
        self.tok2i[tok] = len(self.i2tok)
        self.i2tok.append(tok)

    self.tokenizer = TweetTokenizer()
    print('Dictionary with vocabulary size {} is created'.format(
        len(self.i2tok)))

  def __len__(self):
    return len(self.i2tok)

  def __getitem__(self, tok):
    return self.tok2i.get(tok, self.tok2i[UNK_TOKEN])

  def encode(self, msg,
             include_start=False,
             include_end=False):
    if self.split:
      ret = [self[tok] for tok in split_tokenize(msg.lower())]
    else:
      ret = [self[tok] for tok in self.tokenizer.tokenize(msg.lower())]
    ret = ret + [self[END_TOKEN]] if include_end else ret
    ret = [self[START_TOKEN]] + ret if include_start else ret
    return ret

  def decode(self, toks):
    res = []
    for tok in toks:
      tok = self.i2tok[tok]
      if tok != END_TOKEN:
        res.append(tok)
      else:
        break
    return ' '.join(res)

  def add(self, msg):
    for tok in split_tokenize(msg):
      if tok not in self.tok2i:
        self.tok2cnt[tok] = 0
        self.tok2i[tok] = len(self.i2tok)
        self.i2tok.append(tok)
      self.tok2cnt[tok] += 1

  def save(self, file):
    toklist = [(tok, cnt) for tok, cnt in self.tok2cnt.items()]
    sorted_list = sorted(toklist, key=lambda x: x[1], reverse=True)

    with open(file, 'w') as f:
      for tok in sorted_list:
        if tok[0] not in SPECIALS:
          f.write(tok[0] + '\t' + str(tok[1]) + '\n')
