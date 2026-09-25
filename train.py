"""
Training pipeline for Voice-Enabled Chatbot Intent Classification Model.
Extracts patterns, processes vocabulary, trains the deep neural network,
and serializes artifacts to the model/ directory.
"""

import os
import json
import numpy as np

from src.nlp_utils import tokenize, stem, bag_of_words
from src.model import build_intent_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "intents.json")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "chatbot_model.keras")
WORDS_SAVE_PATH = os.path.join(MODEL_DIR, "words.json")
CLASSES_SAVE_PATH = os.path.join(MODEL_DIR, "classes.json")


def train_model(epochs: int = 150, batch_size: int = 4):
    print("Loading dataset from:", DATA_PATH)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    words = []
    classes = []
    documents = []
    ignore_words = ["?", "!", ".", ",", "'", '"', ";", ":", "-"]

    for intent in data["intents"]:
        tag = intent["tag"]
        if tag not in classes:
            classes.append(tag)
        for pattern in intent["patterns"]:
            tokens = tokenize(pattern)
            words.extend(tokens)
            documents.append((tokens, tag))

    # Stem and remove duplicates & ignored tokens
    words = [stem(w) for w in words if w not in ignore_words and len(w) > 0]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))

    print(f"Total training patterns: {len(documents)}")
    print(f"Unique intent classes: {len(classes)} ({classes})")
    print(f"Unique vocabulary words: {len(words)}")

    # Create training vectors
    x_train = []
    y_train = []

    for pattern_tokens, tag in documents:
        bow = bag_of_words(pattern_tokens, words)
        x_train.append(bow)

        label_row = np.zeros(len(classes), dtype=np.float32)
        label_row[classes.index(tag)] = 1.0
        y_train.append(label_row)

    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Build Deep Learning model
    model = build_intent_model(input_dim=len(words), num_classes=len(classes))
    model.summary()

    # Train model
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        shuffle=True
    )

    final_acc = history.history["accuracy"][-1]
    final_loss = history.history["loss"][-1]
    print(f"Training completed. Final Accuracy: {final_acc:.4f}, Loss: {final_loss:.4f}")

    # Serialize artifacts
    model.save(MODEL_SAVE_PATH)
    print("Model saved to:", MODEL_SAVE_PATH)

    with open(WORDS_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(words, f, indent=2)
    print("Vocabulary saved to:", WORDS_SAVE_PATH)

    with open(CLASSES_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)
    print("Classes saved to:", CLASSES_SAVE_PATH)

    return {
        "final_accuracy": float(final_acc),
        "final_loss": float(final_loss),
        "patterns_count": len(documents),
        "classes_count": len(classes),
        "vocabulary_count": len(words)
    }


if __name__ == "__main__":
    train_model()
