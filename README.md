# EchoBot: Voice-Enabled AI Chatbot

A voice-enabled interactive chatbot powered by Speech Recognition and a Deep Learning Intent Classification Neural Network. Built with Python, Flask, TensorFlow/Keras, and modern Web APIs.

---

## Key Features

- **Voice Input via Speech Recognition:** Real-time speech-to-text processing using the native Web Speech API and backend Python audio processing fallback.
- **Deep Learning Intent Classifier:** Multi-Layer Perceptron (MLP) trained on conversational intent patterns, featuring Dropout regularization and Softmax probability distribution.
- **Dual Display:** Displays both the transcribed user voice query and the generated chatbot response with confidence scoring and intent badges.
- **Text-to-Speech (TTS) Voice Synthesis:** Speaks responses aloud using SpeechSynthesis API, complete with mute/unmute toggle and re-play buttons.
- **Modern Responsive UI:** Glassmorphism dark-theme aesthetics with dynamic microphone recording pulse and audio visualizer wave animations.
- **Self-Contained NLP Pipeline:** Contraction-expansion tokenization and Porter stemming with zero external runtime download dependencies.
- **Cloud-Ready:** Pre-configured for deployment on Render, Railway, or Heroku with health check monitoring.

---

## Local Setup & Execution

### 1. Prerequisites

- Python 3.10, 3.11, 3.12, or 3.13
- Git

### 2. Installation

```bash
# Clone the repository
git clone <your-repository-url>
cd Chatbot

# Install dependencies (using uv or pip)
uv pip install -r requirements.txt
# or: pip install -r requirements.txt
```

### 3. Model Training

```bash
python train.py
```

### 4. Running the Application

```bash
python app.py
```

Open your browser and navigate to: `http://localhost:5000`

---

## Running the Automated Test Suite

```bash
# Run all tests at once
python test/run_all_tests.py

# Or run individual modules:
python test/test_model.py
python test/test_speech.py
python test/test_api.py
```
