"""
Chatbot Inference Engine.
Loads the trained Keras model and predicts intent with confidence scoring,
returning mapped conversational responses with graceful fallback handling.
"""

import os
import json
import random
import numpy as np
import keras

from src.nlp_utils import tokenize, bag_of_words

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "intents.json")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "model", "chatbot_model.keras")
WORDS_SAVE_PATH = os.path.join(BASE_DIR, "model", "words.json")
CLASSES_SAVE_PATH = os.path.join(BASE_DIR, "model", "classes.json")

CONFIDENCE_THRESHOLD = 0.50


class ChatbotEngine:
    def __init__(self):
        self.words = []
        self.classes = []
        self.intents = {}
        self.model = None
        self._load_artifacts()

    def _load_artifacts(self):
        if not os.path.exists(MODEL_SAVE_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_SAVE_PATH}. Please run train.py first."
            )

        with open(WORDS_SAVE_PATH, "r", encoding="utf-8") as f:
            self.words = json.load(f)

        with open(CLASSES_SAVE_PATH, "r", encoding="utf-8") as f:
            self.classes = json.load(f)

        with open(DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.intents = {item["tag"]: item for item in data["intents"]}

        self.model = keras.models.load_model(MODEL_SAVE_PATH)

    def predict_intent(self, text: str):
        """
        Tokenizes text, produces Bag-of-Words, and runs forward pass through the model.
        Returns: (predicted_intent_tag, confidence_score)
        """
        if not text or not text.strip():
            return "fallback", 0.0

        tokens = tokenize(text)
        bow = bag_of_words(tokens, self.words)
        bow_input = np.array([bow], dtype=np.float32)

        predictions = self.model.predict(bow_input, verbose=0)[0]
        max_idx = int(np.argmax(predictions))
        confidence = float(predictions[max_idx])
        predicted_tag = self.classes[max_idx]

        if confidence < CONFIDENCE_THRESHOLD:
            return "fallback", confidence

        return predicted_tag, confidence

    def get_response(self, text: str):
        """
        Processes user query, detects intent, and selects response.
        Returns dict with:
        - query: original user text
        - intent: classified intent
        - confidence: confidence score
        - response: generated response text
        """
        predicted_tag, confidence = self.predict_intent(text)

        if predicted_tag in self.intents:
            responses = self.intents[predicted_tag].get("responses", [])
            response_text = random.choice(responses) if responses else "I hear you!"
        else:
            fallback_responses = self.intents.get("fallback", {}).get(
                "responses", ["I'm sorry, I didn't quite catch that. Could you rephrase?"]
            )
            response_text = random.choice(fallback_responses)

        return {
            "query": text,
            "intent": predicted_tag,
            "confidence": round(confidence, 4),
            "response": response_text
        }


# Global singleton chatbot instance for fast reuse
_chatbot_instance = None


def get_chatbot() -> ChatbotEngine:
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = ChatbotEngine()
    return _chatbot_instance
