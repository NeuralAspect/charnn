import json
import os

import numpy as np
import torch

# /Users/garethdavies/Development/workspaces/charnn/data/raw/input.txt

torch.manual_seed(123)

# data I/O
# TRAINING SET
data = open("data/raw/input_short.txt", "r").read()  # should be simple plain text file

# TOKENIZER
chars = list(set(data))
chars.sort()
data_size, vocab_size = len(data), len(chars)
print("data has %d characters, %d unique." % (data_size, vocab_size))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# Save vocabulary mappings
os.makedirs("models/saved", exist_ok=True)
with open("models/saved/vocab.json", "w") as f:
    json.dump(
        {
            "char_to_ix": char_to_ix,
            "ix_to_char": {
                str(k): v for k, v in ix_to_char.items()
            },  # Convert int keys to strings for JSON
            "chars": chars,
        },
        f,
    )

# hyperparameters
hidden_size = 100  # size of hidden layer of neurons
seq_length = 25  # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
Wxh = torch.randn(hidden_size, vocab_size) * 0.01  # input to hidden
Whh = torch.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
Why = torch.randn(vocab_size, hidden_size) * 0.01  # hidden to output
bh = torch.zeros((hidden_size, 1))  # hidden bias
by = torch.zeros((vocab_size, 1))  # output bias


def lossFun_torch(inputs, targets, hprev):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = hprev.clone().detach()
    loss = 0
    # forward pass
    for t in range(len(inputs)):

        # ONE-HOT ENCODING
        xs[t] = torch.zeros((vocab_size, 1))  # encode in 1-of-k representation
        xs[t][inputs[t]] = 1
        xs[t].shape
        hs[t] = torch.tanh((Wxh @ xs[t]) + (Whh @ hs[t - 1]) + bh)  # hidden state
        ys[t] = Why @ hs[t] + by  # unnormalized log probabilities for next chars
        ps[t] = torch.exp(ys[t]) / torch.sum(
            torch.exp(ys[t])
        )  # probabilities for next chars
        loss += -torch.log(ps[t][targets[t], 0])  # softmax (cross-entropy loss)
        # backward pass: compute gradients going backwards
        dWxh, dWhh, dWhy = (
            torch.zeros_like(Wxh),
            torch.zeros_like(Whh),
            torch.zeros_like(Why),
        )
        dbh, dby = torch.zeros_like(bh), torch.zeros_like(by)
        dhnext = torch.zeros_like(hs[0])

    for t in reversed(range(len(inputs))):
        dy = ps[t].clone().detach()
        dy[
            targets[t]
        ] -= 1  # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
        dWhy += dy @ hs[t].T
        dby += dy
        dh = Why.T @ dy + dhnext  # backprop into h
        dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity
        dbh += dhraw
        dWxh += dhraw @ xs[t].T
        dWhh += dhraw @ hs[t - 1].T
        dhnext = Whh.T @ dhraw

    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        torch.clamp(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients

    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]


def sample_torch(h, seed_ix, n):
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for first time step
    """
    x = torch.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in range(n):
        h = torch.tanh((Wxh @ x) + (Whh @ h) + bh)
        y = Why @ h + by
        p = torch.exp(y) / torch.sum(torch.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.numpy().ravel())
        x = torch.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes


n, p = 0, 0
mWxh, mWhh, mWhy = torch.zeros_like(Wxh), torch.zeros_like(Whh), torch.zeros_like(Why)
mbh, mby = torch.zeros_like(bh), torch.zeros_like(by)  # memory variables for Adagrad
smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # loss at iteration 0

while n < 500_000:
    # prepare inputs (we're sweeping from left to right in steps seq_length long)
    if p + seq_length + 1 >= len(data) or n == 0:
        hprev = torch.zeros((hidden_size, 1))  # reset RNN memory
        p = 0  # go from start of data

    # CHUNK and TOKENIZE
    inputs = [char_to_ix[ch] for ch in data[p : p + seq_length]]
    targets = [char_to_ix[ch] for ch in data[p + 1 : p + seq_length + 1]]

    # sample from the model now and then
    if n % 100 == 0:
        sample_ix = sample_torch(hprev, inputs[0], 200)
        txt = "".join(ix_to_char[ix] for ix in sample_ix)
        print("----\n %s \n----" % (txt,))

    # forward seq_length characters through the net and fetch gradient
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun_torch(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001  # update smoothing loss
    if n % 100 == 0:
        print("iter %d, loss: %f" % (n, smooth_loss))  # print progress

    # perform parameter update with Adagrad
    for param, dparam, mem in zip(
        [Wxh, Whh, Why, bh, by],
        [dWxh, dWhh, dWhy, dbh, dby],
        [mWxh, mWhh, mWhy, mbh, mby],
    ):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

    p += seq_length  # move data pointer
    n += 1  # iteration counter

    # Save model parameters every 1000 iterations
    if n % 1000 == 0:
        model_state = {
            "Wxh": Wxh,  # Already a torch tensor
            "Whh": Whh,  # Already a torch tensor
            "Why": Why,  # Already a torch tensor
            "bh": bh,  # Already a torch tensor
            "by": by,  # Already a torch tensor
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
        }
        torch.save(model_state, f"models/saved/model_state_{n}.pt")
        print(f"Saved model state at iteration {n}")
