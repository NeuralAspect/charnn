import json
import os

import numpy as np
import torch


def load_model_and_vocab(model_path, vocab_path):
    """Load the model parameters and vocabulary mappings."""
    # Load vocabulary
    with open(vocab_path, "r") as f:
        vocab_data = json.load(f)
        char_to_ix = vocab_data["char_to_ix"]
        ix_to_char = {int(k): v for k, v in vocab_data["ix_to_char"].items()}
        chars = vocab_data["chars"]

    # Load model parameters with weights_only=False since we saved the full state
    model_state = torch.load(model_path, weights_only=False)

    # The tensors are already in the correct format
    Wxh = model_state["Wxh"]
    Whh = model_state["Whh"]
    Why = model_state["Why"]
    bh = model_state["bh"]
    by = model_state["by"]

    return Wxh, Whh, Why, bh, by, char_to_ix, ix_to_char, chars


def generate_text(seed_text, model_params, char_to_ix, ix_to_char, max_length=200):
    """Generate text based on the seed text."""
    Wxh, Whh, Why, bh, by = model_params

    # Ensure all model parameters are torch tensors
    # Wxh = torch.tensor(Wxh, dtype=torch.float32)
    # Whh = torch.tensor(Whh, dtype=torch.float32)
    # Why = torch.tensor(Why, dtype=torch.float32)
    # bh = torch.tensor(bh, dtype=torch.float32)
    # by = torch.tensor(by, dtype=torch.float32)

    # Initialize hidden state
    h = torch.zeros((Wxh.shape[0], 1), dtype=torch.float32)

    # Process seed text
    for char in seed_text:
        if char not in char_to_ix:
            print(f"Warning: Character '{char}' not in vocabulary. Skipping.")
            continue
        x = torch.zeros((len(char_to_ix), 1), dtype=torch.float32)
        x[char_to_ix[char]] = 1
        h = torch.tanh((Wxh @ x) + (Whh @ h) + bh)

    # Generate new text
    generated_text = seed_text
    x = torch.zeros((len(char_to_ix), 1), dtype=torch.float32)
    x[char_to_ix[seed_text[-1]]] = 1

    for _ in range(max_length):
        h = torch.tanh((Wxh @ x) + (Whh @ h) + bh)
        y = Why @ h + by
        p = torch.exp(y) / torch.sum(torch.exp(y))
        # Convert to numpy only for the random choice
        p_np = p.detach().numpy().ravel()
        ix = np.random.choice(range(len(char_to_ix)), p=p_np)
        generated_text += ix_to_char[ix]
        x = torch.zeros((len(char_to_ix), 1), dtype=torch.float32)
        x[ix] = 1

    return generated_text


def main():
    # Load the latest model state
    model_dir = "models/saved"
    model_files = [f for f in os.listdir(model_dir) if f.startswith("model_state_")]
    if not model_files:
        print("No saved model found. Please train the model first.")
        return

    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(model_dir, latest_model)
    vocab_path = os.path.join(model_dir, "vocab.json")

    print(f"Loading model from {model_path}")
    model_params = load_model_and_vocab(model_path, vocab_path)
    Wxh, Whh, Why, bh, by, char_to_ix, ix_to_char, chars = model_params

    while True:
        # Get input from user
        seed_text = input("\nEnter seed text (up to 25 characters) or 'quit' to exit: ")
        if seed_text.lower() == "quit":
            break

        if len(seed_text) > 25:
            print("Warning: Input text will be truncated to 25 characters")
            seed_text = seed_text[:25]

        # Generate and print text
        generated = generate_text(
            seed_text, (Wxh, Whh, Why, bh, by), char_to_ix, ix_to_char
        )
        print("\nGenerated text:")
        print(generated)


if __name__ == "__main__":
    main()
