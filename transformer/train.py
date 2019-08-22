import argparse
import time
import collections
import os
import sys
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
import numpy
np = numpy
import matplotlib.pyplot as plt

from models import RNN, GRU
from models import make_model as TRANSFORMER


##############################################################################
#
# ARG PARSING AND EXPERIMENT SETUP
#
##############################################################################

parser = argparse.ArgumentParser(description='PyTorch Penn Treebank Language Modeling')

# Arguments you may need to set to run different experiments in 4.1 & 4.2.
parser.add_argument('--data', type=str, default='data',
                    help='location of the data corpus. We suggest you change the default\
                    here, rather than passing as an argument, to avoid long file paths.')
parser.add_argument('--model', type=str, default='GRU',
                    help='type of recurrent net (RNN, GRU, TRANSFORMER)')
parser.add_argument('--optimizer', type=str, default='SGD_LR_SCHEDULE',
                    help='optimization algo to use; SGD, SGD_LR_SCHEDULE, ADAM')
parser.add_argument('--seq_len', type=int, default=35,
                    help='number of timesteps over which BPTT is performed')
parser.add_argument('--batch_size', type=int, default=20,
                    help='size of one minibatch')
parser.add_argument('--initial_lr', type=float, default=20.0,
                    help='initial learning rate')
parser.add_argument('--hidden_size', type=int, default=1500,
                    help='size of hidden layers. IMPORTANT: for the transformer\
                    this must be a multiple of 16.')
parser.add_argument('--save_best', action='store_true',
                    help='save the model for the best validation performance')
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of hidden layers in RNN/GRU, or number of transformer blocks in TRANSFORMER')

# Other hyperparameters you may want to tune in your exploration
parser.add_argument('--emb_size', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs to stop after')
parser.add_argument('--dp_keep_prob', type=float, default=0.35,
                    help='dropout *keep* probability. drop_prob = 1-dp_keep_prob \
                    (dp_keep_prob=1 means no dropout)')

# Arguments that you may want to make use of / implement more code for
parser.add_argument('--debug', action='store_true')
parser.add_argument('--save_dir', type=str, default='',
                    help='path to save the experimental config, logs, model \
                    This is automatically generated based on the command line \
                    arguments you pass and only needs to be set if you want a \
                    custom dir name')
parser.add_argument('--evaluate', action='store_true',
                    help="use this flag to run on the test set. Only do this \
                    ONCE for each model setting, and only after you've \
                    completed ALL hyperparameter tuning on the validation set.\
                    Note we are not requiring you to do this.")

# DO NOT CHANGE THIS (setting the random seed makes experiments deterministic,
# which helps for reproducibility)
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

args = parser.parse_args()
argsdict = args.__dict__

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# Use the GPU if you have one
if torch.cuda.is_available():
    print("Using the GPU")
    device = torch.device("cuda")
else:
    print("WARNING: You are about to run on cpu, and this will likely run out \
      of memory. \n You can try setting batch_size=1 to reduce memory usage")
    device = torch.device("cpu")


###############################################################################
#
# DATA LOADING & PROCESSING
#
###############################################################################

# HELPER FUNCTIONS
def _read_words(filename):
    with open(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

# Processes the raw data from text files


def ptb_raw_data(data_path=None, prefix="ptb"):
    train_path = os.path.join(data_path, prefix + ".train.txt")
    valid_path = os.path.join(data_path, prefix + ".valid.txt")
    test_path = os.path.join(data_path, prefix + ".test.txt")

    word_to_id, id_2_word = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    return train_data, valid_data, test_data, word_to_id, id_2_word

# Yields minibatches of data


def ptb_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    print("batch_len", batch_len)
    print("num_steps", num_steps)
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    print("epoch_size", epoch_size)
    # if epoch_size == 0:
    #     raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        yield (x, y)


class Batch:
    "Data processing for the transformer. This class adds a mask to the data."

    def __init__(self, x, pad=-1):
        self.data = x
        self.mask = self.make_mask(self.data, pad)

    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."

        def subsequent_mask(size):
            """ helper function for creating the masks. """
            attn_shape = (1, size, size)
            subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
            return torch.from_numpy(subsequent_mask) == 0

        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


# LOAD DATA
print('Loading data from ' + args.data)
raw_data = ptb_raw_data(data_path=args.data)
train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
vocab_size = len(word_to_id)
print('  vocabulary size: {}'.format(vocab_size))


###############################################################################
#
# DEFINE COMPUTATIONS FOR PROCESSING ONE EPOCH
#
###############################################################################

def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.

    This prevents Pytorch from trying to backpropagate into previous input 
    sequences when we use the final hidden states from one mini-batch as the 
    initial hidden states for the next mini-batch.

    Using the final hidden states in this way makes sense when the elements of 
    the mini-batches are actually successive subsequences in a set of longer sequences.
    This is the case with the way we've processed the Penn Treebank dataset.
    """
    if isinstance(h, Variable):
        return h.detach_()
    else:
        return tuple(repackage_hidden(v) for v in h)


def run_epoch(model, data, is_train=False, lr=1.0, _model='TRANSFORMER'):
    """
    One epoch of training/validation (depending on flag is_train).
    """
    if is_train:
        model.train()
    else:
        model.eval()

    if _model != 'TRANSFORMER':
        hidden = model.init_hidden()
        hidden = hidden.to(device)

    # List to store the average loss at each time step
    loss_t = torch.zeros(model.seq_len)

    # LOOP THROUGH MINIBATCHES
    for step, (x, y) in enumerate(ptb_iterator(data, model.batch_size, model.seq_len)):
        if _model == 'TRANSFORMER':
            batch = Batch(torch.from_numpy(x).long().to(device))
            model.zero_grad()
            outputs = model.forward(batch.data, batch.mask).transpose(1, 0)
        else:
            inputs = torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous().to(device)  # .cuda()
            model.zero_grad()
            # Initialize the initial hidden(h_0) as zero for every mini-batch of
            # size(batch_size, seq_len) for computing the average loss at each time
            # step
            hidden = repackage_hidden(model.init_hidden())

            outputs, hidden = model(inputs, hidden)

        # targets size is (seq_len, batch_size)
        targets = torch.from_numpy(y.astype(np.int64)).transpose(
            0, 1).contiguous().to(device)

        outputs = outputs.transpose(1, 2).contiguous()  # (seq_len, vocab_size, batch_size)

        # LOSS COMPUTATION
        # This line computes average loss across all the sequences in a mini-batch at each time-step separately.
        loss = loss_fn(outputs, targets)  # size of loss is (seq_len, batch_size)
        loss_t += loss.sum(1).detach()

    print('Total no. of tokens at each time step : {}'.format(model.batch_size * (step + 1)))
    loss_t = loss_t / (model.batch_size * (step + 1))

    return loss_t.numpy()


###############################################################################
#
# RUN MAIN LOOP (TRAIN AND VAL)
#
###############################################################################

print("\n########## Running Main Loop ##########################")
# MAIN LOOP
print('Length of Valid data: {}'.format(len(valid_data)))

plt.figure()
for _model in ['RNN', 'GRU', 'TRANSFORMER']:

    if _model == 'RNN':
        model = RNN(emb_size=args.emb_size, hidden_size=args.hidden_size,
                    seq_len=args.seq_len, batch_size=args.batch_size,
                    vocab_size=vocab_size, num_layers=args.num_layers,
                    dp_keep_prob=args.dp_keep_prob)
    elif _model == 'GRU':
        model = GRU(emb_size=args.emb_size, hidden_size=args.hidden_size,
                    seq_len=args.seq_len, batch_size=args.batch_size,
                    vocab_size=vocab_size, num_layers=args.num_layers,
                    dp_keep_prob=args.dp_keep_prob)
    elif _model == 'TRANSFORMER':
        if args.debug:  # use a very small model
            model = TRANSFORMER(vocab_size=vocab_size, n_units=16, n_blocks=2)
        else:
            # Note that we're using num_layers and hidden_size to mean slightly
            # different things here than in the RNNs.
            # Also, the Transformer also has other hyperparameters
            # (such as the number of attention heads) which can change it's behavior.
            transformer_hidden_size = 512
            transformer_num_layers = 6
            model = TRANSFORMER(vocab_size=vocab_size, n_units=transformer_hidden_size,
                                n_blocks=transformer_num_layers, dropout=1. - args.dp_keep_prob)
        # these 3 attributes don't affect the Transformer's computations;
        # they are only used in run_epoch
        model.batch_size = 128
        model.seq_len = 35
        model.vocab_size = vocab_size
    else:
        print("Model type not recognized.")

    # Load the saved best model
    model.load_state_dict(torch.load('best_models/' + _model + '/best_params.pt',
                                     map_location=lambda storage, loc: storage))
    model = model.to(device)

    # LOSS FUNCTION
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    # RUN MODEL ON VALIDATION DATA
    loss_t = run_epoch(model, valid_data, _model=_model)

    plt.plot(loss_t, label=_model)

plt.xlabel('Time steps')
plt.ylabel('Loss at t')
plt.legend()
plt.title('The average loss at each time-step')
plt.savefig('plots/loss_vs_t_.png')
plt.close()
