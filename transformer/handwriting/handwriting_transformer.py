import torch
import math, sys
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import bernoulli, uniform
from utils.model_utils import stable_softmax, subsequent_mask

sys.path.append("../")
from transformer import Encoder, Decoder, positional_embedding


def sample_from_out_dist(y_hat, bias):
    split_sizes = [1] + [20] * 6
    y = torch.split(y_hat, split_sizes, dim=0)

    eos_prob = torch.sigmoid(y[0])
    mixture_weights = stable_softmax(y[1] * (1 + bias), dim=0)
    mu_1 = y[2]
    mu_2 = y[3]
    std_1 = torch.exp(y[4] - bias)
    std_2 = torch.exp(y[5] - bias)
    correlations = torch.tanh(y[6])

    bernoulli_dist = bernoulli.Bernoulli(probs=eos_prob)
    eos_sample = bernoulli_dist.sample()

    K = torch.multinomial(mixture_weights, 1)

    mu_k = y_hat.new_zeros(2)

    mu_k[0] = mu_1[K]
    mu_k[1] = mu_2[K]
    cov = y_hat.new_zeros(2, 2)
    cov[0, 0] = std_1[K].pow(2)
    cov[1, 1] = std_2[K].pow(2)
    cov[0, 1], cov[1, 0] = (
        correlations[K] * std_1[K] * std_2[K],
        correlations[K] * std_1[K] * std_2[K],
    )

    x = torch.normal(mean=torch.Tensor([0.0, 0.0]), std=torch.Tensor([1.0, 1.0])).to(
        y_hat.device
    )
    Z = mu_k + torch.mv(cov, x)

    sample = y_hat.new_zeros(1, 1, 3)
    sample[0, 0, 0] = eos_sample.item()
    sample[0, 0, 1:] = Z
    return sample


class HandwritingTransformer(nn.Module):
    def __init__(
        self,
        src_vocab=77,
        seq_dim=3,
        out_dim=121,
        d_model=512,
        n_heads=8,
        n_enc_layers=6,
        n_dec_layers=6,
    ):
        super(HandwritingTransformer, self).__init__()
        self.src_vocab = src_vocab
        self.out_dim = out_dim
        self.seq_dim = seq_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.n_enc_layers = n_enc_layers
        self.n_dec_layers = n_dec_layers
        self.EOS = False
        self.src_embed_layer = nn.Embedding(self.src_vocab, self.d_model)
        self.tgt_linear = nn.Linear(self.seq_dim, self.d_model)
        self.encoder = Encoder(n_enc_layers, d_model, n_heads, self.d_k)
        self.decoder = Decoder(n_dec_layers, d_model, n_heads, self.d_k)
        self.output_layer = nn.Linear(self.d_model, self.out_dim)
        self.init_weights()

    def init_weights(self):
        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

    def forward(self, text, text_mask, stroke, stroke_mask):
        inp_embed = math.sqrt(self.d_model) * self.src_embed_layer(
            text
        ) + positional_embedding(text, self.d_model)
        target_embed = math.sqrt(self.d_model) * self.tgt_linear(
            stroke
        ) + positional_embedding(stroke, self.d_model)
        encoding = self.encoder(inp_embed, text_mask)
        decoding = self.decoder(encoding, text_mask, target_embed, stroke_mask)
        y_hat = self.output_layer(decoding)
        return y_hat

    def generate(self, text, text_mask, stroke, stroke_mask, bias):
        seq_len = 0
        gen_seq = []
        with torch.no_grad():
            batch_size = stroke.shape[0]
            print("batch_size:", batch_size)
            while seq_len < 2000:
                y_hat = self.forward(text, text_mask, stroke, stroke_mask)
                y_hat = y_hat[-1, -1]
                Z = sample_from_out_dist(y_hat, bias)
                stroke = torch.cat((stroke, Z), dim=1)
                stroke_mask = subsequent_mask(size=stroke.size(1))
                gen_seq.append(Z)
                seq_len += 1

        gen_seq = torch.cat(gen_seq, dim=1)
        gen_seq = gen_seq.cpu().numpy()

        print("seq_len:", seq_len)

        return gen_seq

