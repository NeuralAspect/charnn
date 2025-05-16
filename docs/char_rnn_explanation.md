# Character-Level RNN Implementation Explanation

This document provides a detailed explanation of the character-level Recurrent Neural Network (RNN) implementation in `char_rnn_1.py`.

## Overview

The implementation is a character-level language model that uses a simple RNN architecture to predict the next character in a sequence. It's built using PyTorch and includes both training and sampling functionality.

## Data Preparation

```python
data = open("data/raw/input.txt", "r").read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
```

1. The code reads a text file from `data/raw/input.txt`
2. Creates a vocabulary by getting unique characters from the text
3. Establishes mappings between characters and their indices:
   - `char_to_ix`: maps characters to their numerical indices
   - `ix_to_char`: maps numerical indices back to characters

## Model Architecture

### Hyperparameters
```python
hidden_size = 100  # size of hidden layer of neurons
seq_length = 25    # number of steps to unroll the RNN for
learning_rate = 1e-1
```

### Model Parameters
```python
Wxh = torch.randn(hidden_size, vocab_size) * 0.01  # input to hidden
Whh = torch.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
Why = torch.randn(vocab_size, hidden_size) * 0.01   # hidden to output
bh = torch.zeros((hidden_size, 1))                  # hidden bias
by = torch.zeros((vocab_size, 1))                   # output bias
```

The model consists of:
- Input layer (vocab_size dimensions)
- Hidden layer (100 neurons)
- Output layer (vocab_size dimensions)

## Core Functions

### 1. Loss Function (`lossFun_torch`)

```python
def lossFun_torch(inputs, targets, hprev):
```

This function implements:
1. Forward pass:
   - Converts input characters to one-hot vectors
   - Computes hidden states using tanh activation
   - Calculates output probabilities using softmax
   - Computes cross-entropy loss

2. Backward pass:
   - Computes gradients for all parameters
   - Implements backpropagation through time (BPTT)
   - Includes gradient clipping to prevent exploding gradients

### 2. Sampling Function (`sample_torch`)

```python
def sample_torch(h, seed_ix, n):
```

This function:
1. Takes a seed character and generates n new characters
2. Uses the trained model to predict the next character
3. Samples from the probability distribution to generate text
4. Returns the sequence of generated character indices

## Training Loop

The main training loop:
1. Processes the input data in sequences of length `seq_length`
2. Resets the hidden state when reaching the end of the data
3. Performs forward and backward passes
4. Updates parameters using Adagrad optimization
5. Periodically samples from the model to show progress

Key features:
- Uses Adagrad optimization for parameter updates
- Implements gradient clipping
- Maintains a smoothed loss for monitoring
- Samples from the model every 100 iterations

## Usage

To use this model:
1. Place your training text in `data/raw/input.txt`
2. Run the script
3. The model will train and periodically show generated samples
4. Training continues until manually stopped

## Technical Details

### Memory Management
- Uses PyTorch tensors for efficient computation
- Implements proper gradient handling with `detach()`
- Manages memory states for the RNN

### Optimization
- Uses Adagrad optimization
- Implements gradient clipping (-5 to 5)
- Learning rate of 0.1

### Sampling
- Uses temperature-based sampling
- Converts model outputs to probabilities using softmax
- Samples from the probability distribution to generate text

## Notes

- The model is relatively simple but effective for character-level language modeling
- Training can be computationally intensive
- The quality of generated text depends on:
  - The training data
  - The model's hyperparameters
  - The training duration 