import numpy as np
import torch

# data I/O
data = open("data/raw/input.txt", "r").read()  # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print("data has %d characters, %d unique." % (data_size, vocab_size))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# hyperparameters
hidden_size = 100  # size of hidden layer of neurons
seq_length = 25  # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
Wxh = (torch.randn(hidden_size, vocab_size) * 0.01).requires_grad_()  # input to hidden
Whh = (
    torch.randn(hidden_size, hidden_size) * 0.01
).requires_grad_()  # hidden to hidden
Why = (torch.randn(vocab_size, hidden_size) * 0.01).requires_grad_()  # hidden to output
bh = torch.zeros((hidden_size, 1), requires_grad=True)  # hidden bias
by = torch.zeros((vocab_size, 1), requires_grad=True)  # output bias

optimizer = torch.optim.Adagrad([Wxh, Whh, Why, bh, by], lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()


def lossFun_torch(inputs, targets, hprev):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = hprev.clone().detach()
    optimizer.zero_grad()

    # forward pass
    for t in range(len(inputs)):
        xs[t] = torch.zeros((vocab_size, 1))  # encode in 1-of-k representation
        xs[t][inputs[t]] = 1
        hs[t] = torch.tanh((Wxh @ xs[t]) + (Whh @ hs[t - 1]) + bh)  # hidden state
        ys[t] = Why @ hs[t] + by  # unnormalized log probabilities for next chars

    ys = torch.stack(list(ys.values())).squeeze()
    targets = torch.tensor(targets)
    loss = loss_fn(ys, targets)
    loss.backward()
    optimizer.step()

    return loss, hs[len(inputs) - 1]


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
        ix = np.random.choice(range(vocab_size), p=p.detach().numpy().ravel())
        x = torch.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes


n, p = 0, 0
smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # loss at iteration 0

while True:
    # prepare inputs (we're sweeping from left to right in steps seq_length long)
    if p + seq_length + 1 >= len(data) or n == 0:
        hprev = torch.zeros((hidden_size, 1))  # reset RNN memory
        p = 0  # go from start of data

    inputs = [char_to_ix[ch] for ch in data[p : p + seq_length]]
    targets = [char_to_ix[ch] for ch in data[p + 1 : p + seq_length + 1]]

    # sample from the model now and then
    if n % 100 == 0:
        sample_ix = sample_torch(hprev, inputs[0], 200)
        txt = "".join(ix_to_char[ix] for ix in sample_ix)
        print("----\n %s \n----" % (txt,))

    # forward seq_length characters through the net and fetch gradient
    loss, hprev = lossFun_torch(inputs, targets, hprev)

    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n % 100 == 0:
        print("iter %d, loss: %f" % (n, smooth_loss))  # print progress

    p += seq_length  # move data pointer
    n += 1  # iteration counter
