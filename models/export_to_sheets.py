import json

import numpy as np
import pandas as pd
import torch


def export_model_to_sheets(model_path, vocab_path, output_dir):
    """Export model weights and vocabulary to CSV files for Google Sheets."""
    # Load the model and vocabulary
    with open(vocab_path, "r") as f:
        vocab_data = json.load(f)
        char_to_ix = vocab_data["char_to_ix"]
        ix_to_char = {int(k): v for k, v in vocab_data["ix_to_char"].items()}
        chars = vocab_data["chars"]

    model_state = torch.load(model_path, weights_only=False)

    # Convert model parameters to numpy arrays
    Wxh = model_state["Wxh"].numpy()
    Whh = model_state["Whh"].numpy()
    Why = model_state["Why"].numpy()
    bh = model_state["bh"].numpy()
    by = model_state["by"].numpy()

    # Create DataFrames for each parameter
    Wxh_df = pd.DataFrame(Wxh)
    Whh_df = pd.DataFrame(Whh)
    Why_df = pd.DataFrame(Why)
    bh_df = pd.DataFrame(bh)
    by_df = pd.DataFrame(by)

    # Save parameters to CSV
    Wxh_df.to_csv(f"{output_dir}/Wxh.csv", index=False, header=False)
    Whh_df.to_csv(f"{output_dir}/Whh.csv", index=False, header=False)
    Why_df.to_csv(f"{output_dir}/Why.csv", index=False, header=False)
    bh_df.to_csv(f"{output_dir}/bh.csv", index=False, header=False)
    by_df.to_csv(f"{output_dir}/by.csv", index=False, header=False)

    # Save vocabulary
    vocab_df = pd.DataFrame(
        {
            "char": [ix_to_char[i] for i in range(len(chars))],
            "index": list(range(len(chars))),
        }
    )
    vocab_df.to_csv(f"{output_dir}/vocab.csv", index=False)

    # Create a README with instructions
    with open(f"{output_dir}/README.txt", "w") as f:
        f.write(
            """Google Sheets Character RNN Model

Files:
- Wxh.csv: Input to hidden weights
- Whh.csv: Hidden to hidden weights
- Why.csv: Hidden to output weights
- bh.csv: Hidden bias
- by.csv: Output bias
- vocab.csv: Character to index mapping

Instructions for Google Sheets:
1. Import each CSV file into separate sheets
2. For one-hot encoding, use: =IF(COLUMN()=B2+1,1,0) where B2 is the character index
3. For matrix multiplication, use: =MMULT()
4. For tanh, use: =TANH()
5. For softmax, use: =EXP(A1)/SUM(EXP(A1:A10)) where A1:A10 is your range
6. For random sampling, use: =RANDBETWEEN(1,100)/100

Note: You'll need to implement the full RNN logic using these basic operations.
"""
        )


if __name__ == "__main__":
    import os

    os.makedirs("models/sheets_export", exist_ok=True)
    export_model_to_sheets(
        "models/saved/model_state_99000.pt",
        "models/saved/vocab.json",
        "models/sheets_export",
    )
