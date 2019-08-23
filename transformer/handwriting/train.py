import torch
import math
import os
import torch.nn as nn
import numpy as np
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributions import bernoulli, uniform
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from handwriting_transformer import HandwritingTransformer
from utils import plot_stroke
from utils.constants import Global
from utils.dataset import HandwritingDataset
from utils.model_utils import compute_nll_loss, subsequent_mask
from utils.data_utils import data_denormalization


def generate_conditional_sequence(
    model, char_seq, device, char_to_id, idx_to_char, bias
):

    model.eval()
    stroke = torch.zeros(1, 1, 3).to(device)
    stroke_mask = subsequent_mask(size=1).int()
    char_seq = np.array(list(char_seq + "  "))
    print("".join(char_seq))
    text = np.array([[char_to_id[char] for char in char_seq]]).astype(np.long)
    print("text:", text)
    text = torch.from_numpy(text).to(device)
    text_mask = torch.ones(text.shape).to(device)

    print("Generating sequence....")
    gen_seq = model.generate(text, text_mask, stroke, stroke_mask, bias)

    length = len(text_mask.nonzero())
    print("Input seq: ", "".join(idx_to_char(text[0].detach().cpu().numpy()))[:length])
    print("Length of input sequence: ", text[0].shape[0])

    return gen_seq


def argparser():

    parser = argparse.ArgumentParser(description="PyTorch Handwriting Synthesis Model")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--data_path", type=str, default="./data/")
    parser.add_argument("--save_path", type=str, default="./logs/")
    parser.add_argument("--text_req", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--data_aug", action="store_true")
    parser.add_argument("--seed", type=int, default=212, help="random seed")
    args = parser.parse_args()

    return args


def train_epoch(model, optimizer, epoch, train_loader, device):
    avg_loss = 0.0
    model.train()
    for i, mini_batch in enumerate(train_loader):

        stroke, targets, mask, std_mask, text, text_mask = mini_batch
        text = text.to(device)
        text_mask = text_mask.to(device)

        stroke = stroke.to(device)
        targets = targets.to(device)
        mask = mask.to(device)
        std_mask = std_mask.to(device)

        batch_size = stroke.shape[0]

        optimizer.zero_grad()

        y_hat = model.forward(text, text_mask, stroke, std_mask)

        loss = compute_nll_loss(targets, y_hat, mask)

        # Output gradient clipping
        y_hat.register_hook(lambda grad: torch.clamp(grad, -100, 100))

        loss.backward()

        optimizer.step()
        avg_loss += loss.item()

        # print every 10 mini-batches
        if i % 10 == 0:
            print(
                "[{:d}, {:5d}] loss: {:.3f}".format(epoch + 1, i + 1, loss / batch_size)
            )
    avg_loss /= len(train_loader.dataset)

    return avg_loss


def validation(model, valid_loader, device, epoch):
    avg_loss = 0.0
    model.eval()

    with torch.no_grad():
        for i, mini_batch in enumerate(valid_loader):

            stroke, targets, mask, std_mask, text, text_mask = mini_batch
            text = text.to(device)
            text_mask = text_mask.to(device)

            stroke = stroke.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            std_mask = std_mask.to(device)
            batch_size = stroke.shape[0]

            y_hat = model.forward(text, text_mask, stroke, std_mask)

            loss = compute_nll_loss(targets, y_hat, mask)
            avg_loss += loss.item()

            # print every 10 mini-batches
            if i % 10 == 0:
                print(
                    "[{:d}, {:5d}] loss: {:.3f}".format(
                        epoch + 1, i + 1, loss / batch_size
                    )
                )

    avg_loss /= len(valid_loader.dataset)

    return avg_loss


def train(
    model,
    train_loader,
    valid_loader,
    batch_size,
    n_epochs,
    lr,
    patience,
    device,
    save_path,
):
    model_path = save_path + "best_model.pt"
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    valid_losses = []
    best_loss = math.inf
    best_epoch = 0
    k = 0
    for epoch in range(n_epochs):
        print("training.....")
        train_loss = train_epoch(model, optimizer, epoch, train_loader, device)

        print("validation....")
        valid_loss = validation(model, valid_loader, device, epoch)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print("Epoch {}: Train: avg. loss: {:.3f}".format(epoch + 1, train_loss))
        print("Epoch {}: Valid: avg. loss: {:.3f}".format(epoch + 1, valid_loss))

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch + 1
            print("Saving best model at epoch {}".format(epoch + 1))
            torch.save(model.state_dict(), model_path)

            gen_seq = generate_conditional_sequence(
                model,
                "Hello world!",
                device,
                train_loader.dataset.char_to_id,
                train_loader.dataset.idx_to_char,
                bias=10.0,
            )

            # denormalize the generated offsets using train set mean and std
            gen_seq = data_denormalization(Global.train_mean, Global.train_std, gen_seq)

            # plot the sequence
            plot_stroke(
                gen_seq[0], save_name=save_path + "_seq_" + str(best_epoch) + ".png"
            )
            k = 0
        elif k > patience:
            print("Best model was saved at epoch: {}".format(best_epoch))
            print("Early stopping at epoch {}".format(epoch))
            break
        else:
            k += 1


if __name__ == "__main__":

    args = argparser()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # fix random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Arguments: {}".format(args))
    batch_size = args.batch_size
    n_epochs = args.n_epochs

    # Load the data and text
    train_dataset = HandwritingDataset(
        args.data_path,
        split="train",
        text_req=args.text_req,
        debug=args.debug,
        data_aug=args.data_aug,
    )
    valid_dataset = HandwritingDataset(
        args.data_path,
        split="valid",
        text_req=args.text_req,
        debug=args.debug,
        data_aug=args.data_aug,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = HandwritingTransformer(
        src_vocab=train_dataset.vocab_size,
        seq_dim=3,
        out_dim=121,
        d_model=512,
        n_heads=8,
        n_enc_layers=6,
        n_dec_layers=6,
    )

    train(
        model,
        train_loader,
        valid_loader,
        batch_size,
        n_epochs,
        args.lr,
        args.patience,
        device,
        args.save_path,
    )

