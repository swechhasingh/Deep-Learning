import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(
        self, vocab_size, d_model=512, n_heads=8, n_encoder_layers=6, n_decoder_layers=6
    ):
        super(Transformer, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers

        self.embedding_layer = math.sqrt(d_model) * nn.Linear(
            self.vocab_size, self.d_model
        )
        self.output_layer = nn.Sequential(
            [nn.Linear(self.d_model, self.vocab_size), nn.Softmax(dim=2)]
        )

    def forward(self, inp):
        embedding = self.embedding_layer(inp) + self.positional_embedding(inp)
        encoding = self.encoder(embedding)
        decoder_output = self.decoder(encoding)
        return decoder_output

    def positional_embedding(self, inp):
        pos_embed = inp.new_zeros((inp.shape[0], inp.shape[1], self.d_model))
        for i in range(self.d_model // 2):
            for pos in range(pos_embed.shape[1]):
                x = pos / (10000.0).pow(2 * i / self.d_model)
                pos_embed[:, pos, 2 * i] = torch.sin(x)
                pos_embed[:, pos, 2 * i + 1] = torch.cos(x)
        return pos_embed

    def pointwise_feed_forward_layer(self, d_ff):
        return nn.Sequential(
            [nn.Linear(self.d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, self.d_model)]
        )

    def encoder(self,):

        pass

    def decoder(self,):

        pass

    def attention(self, query, key, value, is_mask=False):
        """
            Scaled dot-product attention
        """
        dot_prod = torch.matmul(query, torch.transpose(key, 1, 2))
        scale_dot_prod = (1.0 / math.sqrt(self.d_k)) * dot_prod
        if is_mask:
            mask = torch.tril(
                torch.zeros(query.shape[0], query.shape[1]).to(query.device)
            )
        else:
            mask = 1.0
        maked_dot_prod = mask * scale_dot_prod
        weights = F.softmax(maked_dot_prod, dim=2)
        attention = torch.matmul(weights, value)  # nxtxd_k
        return attention

    def multi_head_attention(self, query, key, value, is_mask=False):
        """
            Multi-Head Attention
        """
        q_linear = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_k) for _ in range(self.n_heads)]
        )
        k_linear = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_k) for _ in range(self.n_heads)]
        )
        v_linear = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_k) for _ in range(self.n_heads)]
        )
        heads = [
            self.attention(
                q_linear[i](query), k_linear[i](key), v_linear[i](value), is_mask
            )
            for i in range(self.n_heads)
        ]
        heads = torch.cat(heads, dim=2)
        linear_layer = nn.Linear(self.d_k * self.n_heads, self.d_model)

        return linear_layer(heads)
