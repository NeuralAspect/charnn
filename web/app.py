import os
import sys

from flask import Flask, jsonify, render_template, request

# Add the models directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.inference import generate_text, load_model_and_vocab

app = Flask(__name__)

# Load the model when the app starts
model_dir = os.path.join(os.path.dirname(__file__), "..", "models", "saved")
model_files = [f for f in os.listdir(model_dir) if f.startswith("model_state_")]
if not model_files:
    raise Exception("No saved model found. Please train the model first.")

latest_model = sorted(model_files)[-1]
model_path = os.path.join(model_dir, latest_model)
vocab_path = os.path.join(model_dir, "vocab.json")

print(f"Loading model from {model_path}")
model_params = load_model_and_vocab(model_path, vocab_path)
Wxh, Whh, Why, bh, by, char_to_ix, ix_to_char, chars = model_params


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    seed_text = data.get("text", "")

    if len(seed_text) > 25:
        seed_text = seed_text[:25]

    try:
        generated = generate_text(
            seed_text, (Wxh, Whh, Why, bh, by), char_to_ix, ix_to_char
        )
        return jsonify({"generated_text": generated})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
