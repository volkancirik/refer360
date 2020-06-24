import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from src.utils import PositionalEncoding


class VisualBert(nn.Module):
  """
  Vision only unimodal baseline
  """

  def __init__(self, args, n_vocab,
               hc1=512, k=5):
    super(VisualBert, self).__init__()

    self.n_vocab = n_vocab
    self.n_emb = args.n_emb
    self.n_head = args.n_head
    self.n_hid = args.n_hid
    self.n_layers = args.n_layers
    self.dropout = args.dropout

    self.pos_encoder = PositionalEncoding(self.n_emb, self.dropout)
    encoder_layers = TransformerEncoderLayer(
        self.n_emb, self.n_head, self.n_hid, self.dropout)
    self.transformer_encoder = TransformerEncoder(
        encoder_layers, self.n_layers)
    self.encoder = nn.Embedding(self.n_vocab, self.n_emb)

    self.act = nn.LeakyReLU()
    self.deconv1 = nn.ConvTranspose2d(
        hc1, hc1, k, stride=3, padding=2)
    self.deconv2 = nn.ConvTranspose2d(
        hc1, hc1, k, stride=2, padding=2)
    self.deconv3 = nn.ConvTranspose2d(
        hc1, hc1, k, stride=2, padding=2)
    self.deconv4 = nn.ConvTranspose2d(
        hc1, hc1, k, stride=2, padding=1)
    self.deconv5 = nn.ConvTranspose2d(
        hc1, hc1, k, stride=2, padding=2)
    self.deconv6 = nn.ConvTranspose2d(
        hc1, 1, k, stride=2, padding=2)
    self.init_weights()

  def init_weights(self):

    self.deconv1.bias.data.fill_(0)
    self.deconv2.bias.data.fill_(0)
    self.deconv3.bias.data.fill_(0)
    self.deconv4.bias.data.fill_(0)
    self.deconv5.bias.data.fill_(0)

  def forward(self, images, texts, seq_lengths,
              texts_mask=None,
              texts_emb_mask=None):

    batch_size = texts_mask.size(0)
    text_encoding = self.encoder(texts)

    image_mask = torch.ones(batch_size, 169).cuda().bool()
    image_emb_mask = torch.ones(169, batch_size, 512).cuda()
    image_encoding = images.reshape(batch_size, 512, 169).permute(2, 0, 1)

    fused_mask = torch.cat([image_mask, texts_mask], dim=1)
    fused_emb_mask = torch.cat([image_emb_mask, texts_emb_mask], dim=0)
    fused_input = torch.cat([image_encoding, text_encoding], dim=0)

    fused_src = self.pos_encoder(fused_input)
    fused_encoding = self.transformer_encoder(
        fused_src, src_key_padding_mask=fused_mask)
    fused_encoding = fused_encoding * fused_emb_mask
    fused_encoding = torch.sum(fused_encoding, dim=0)
    deconv_input = fused_encoding.unsqueeze(2).unsqueeze(3)

    d1 = self.act(self.deconv1(
        deconv_input, output_size=(batch_size, 512, 3, 3)))
    d2 = self.act(self.deconv2(
        d1, output_size=(batch_size, 512, 6, 6)))
    d3 = self.act(self.deconv3(
        d2, output_size=(batch_size, 512, 12, 12)))
    d4 = self.act(self.deconv4(
        d3, output_size=(batch_size, 512, 25, 25)))
    d5 = self.act(self.deconv5(
        d4, output_size=(batch_size, 512, 50, 50)))
    d6 = self.deconv6(
        d5, output_size=(batch_size, 512, 100, 100)).squeeze(1)
    out = F.log_softmax(d6.view(batch_size, -1),
                        1).view(batch_size, 100, 100)
    return out
