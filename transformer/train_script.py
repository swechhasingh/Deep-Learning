import torch
import numpy as np
import math, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
from transformer import Transformer, positional_embedding, attention


def subsequent_mask(size):
    "Mask out subsequent positions."
    """
        Args: size: length of sentences
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(subsequent_mask) == 0


# Batching and masking
class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)  # nx1xl
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)  # nx1xl
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        )
        return tgt_mask


def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print(
                "Epoch Step: %d Loss: %f Tokens per Sec: %f"
                % (i, loss / batch.ntokens, tokens / elapsed)
            )
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


if __name__ == "__main__":
    plt.figure(figsize=(5, 5))
    plt.imshow(subsequent_mask(20)[0])
    plt.show()
    print(subsequent_mask(20)[0])
    model = Transformer(
        src_vocab=10,
        tgt_vocab=10,
        d_model=512,
        n_heads=8,
        n_enc_layers=6,
        n_dec_layers=6,
    )
    print(model)
