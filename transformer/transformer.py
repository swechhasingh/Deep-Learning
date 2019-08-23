import torch
import math, copy
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def attention(d_k, query, key, value, mask=None):
    """
        Scaled dot-product attention
    """
    dot_prod = torch.matmul(query, torch.transpose(key, 1, 2))
    scale_dot_prod = (1.0 / math.sqrt(d_k)) * dot_prod
    if mask is not None:
        scale_dot_prod = scale_dot_prod.masked_fill(mask == 0, -1e9)
    # attention
    weights = F.softmax(scale_dot_prod, dim=-1)
    new_value = torch.matmul(weights, value)  # nxtxd_k
    return new_value, weights


def positional_embedding(inp, d_model, max_len=2000):
    pos_embed = inp.new_zeros((max_len, d_model)).float()
    pos = torch.arange(max_len).float().unsqueeze(1)  # max_lenx1
    exp_term = torch.exp(
        math.log(10000.0) * (-torch.arange(0, d_model, 2).float() / d_model)
    )  # d_model
    x = pos * exp_term  # max_len x (d_model // 2)
    pos_embed[:, 0::2] = torch.sin(x)
    pos_embed[:, 1::2] = torch.cos(x)
    pos_embed = pos_embed.unsqueeze(0)  # (1, max_len, self.d_model)

    return Variable(pos_embed[:, : inp.shape[1]], requires_grad=False)


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab,
        tgt_vocab=None,
        d_model=512,
        n_heads=8,
        n_enc_layers=6,
        n_dec_layers=6,
    ):
        super(Transformer, self).__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.n_enc_layers = n_enc_layers
        self.n_dec_layers = n_dec_layers

        self.src_embed_layer = nn.Embedding(self.src_vocab, self.d_model)
        self.tgt_embed_layer = nn.Embedding(self.tgt_vocab, self.d_model)
        self.encoder = Encoder(n_enc_layers, d_model, n_heads, self.d_k)
        self.decoder = Decoder(n_dec_layers, d_model, n_heads, self.d_k)
        self.output_layer = nn.Linear(self.d_model, self.tgt_vocab)
        self.init_weights()

    def init_weights(self):
        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

    def forward(self, inp, inp_mask, tgt, tgt_mask):
        inp_embed = math.sqrt(self.d_model) * self.src_embed_layer(
            inp
        ) + positional_embedding(inp, self.d_model)
        target_embed = math.sqrt(self.d_model) * self.tgt_embed_layer(
            tgt
        ) + positional_embedding(tgt, self.d_model)
        encoding = self.encoder(inp_embed, inp_mask)
        decoding = self.decoder(encoding, inp_mask, target_embed, tgt_mask)
        logits = self.output_layer(decoding)
        return logits


class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_k):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.layers = nn.ModuleList(
            [EncoderLayer(n_heads, d_model, d_k) for _ in range(n_layers)]
        )

    def forward(self, x, mask):
        for sub_layer in self.layers:
            x = sub_layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_k):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.layers = nn.ModuleList(
            [DecoderLayer(n_heads, d_model, d_k) for _ in range(n_layers)]
        )

    def forward(self, encoding, src_mask, target, target_mask):
        for sub_layer in self.layers:
            target = sub_layer(encoding, src_mask, target, target_mask)
        return target


class PointWiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, p=0.5):
        super(PointWiseFeedForwardNetwork, self).__init__()
        self.d_ff = d_ff
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.linear_2(self.relu(x))
        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_ff=2048, p=0.1):
        super(EncoderLayer, self).__init__()
        self.attn_layer = MultiHeadAttention(n_heads, d_model, d_k, p=0.1)
        self.ff_layer = PointWiseFeedForwardNetwork(d_model, d_ff, p=0.5)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p)

    def forward(self, x, mask):
        x = self.norm(x + self.dropout(self.attn_layer(x, x, x, mask)))
        return self.norm(x + self.dropout(self.ff_layer(x)))


class DecoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_ff=2048, p=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_heads, d_model, d_k, p=0.1)
        self.enc_dec_attn = MultiHeadAttention(n_heads, d_model, d_k, p=0.1)
        self.ff_layer = PointWiseFeedForwardNetwork(d_model, d_ff, p=0.5)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p)

    def forward(self, encoding, src_mask, target, target_mask):
        target = self.norm(
            target + self.dropout(self.self_attn(target, target, target, target_mask))
        )
        target = self.norm(
            target
            + self.dropout(self.enc_dec_attn(target, encoding, encoding, src_mask))
        )
        return self.norm(target + self.dropout(self.ff_layer(target)))


class MultiHeadAttention(nn.Module):
    """
        Multi-Head Attention
    """

    def __init__(self, n_heads, d_model, d_k, p=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.n_heads = n_heads
        self.q_linear = nn.ModuleList([nn.Linear(d_model, d_k) for _ in range(n_heads)])
        self.k_linear = nn.ModuleList([nn.Linear(d_model, d_k) for _ in range(n_heads)])
        self.v_linear = nn.ModuleList([nn.Linear(d_model, d_k) for _ in range(n_heads)])
        self.linear_layer = nn.Linear(self.d_k * self.n_heads, d_model)
        self.dropout = nn.Dropout(p)

    def forward(self, query, key, value, mask=None):

        heads = [
            attention(
                self.d_k,
                self.q_linear[i](query),
                self.k_linear[i](key),
                self.v_linear[i](value),
                mask,
            )[0]
            for i in range(self.n_heads)
        ]
        heads = torch.cat(heads, dim=-1)

        return self.dropout(self.linear_layer(heads))
