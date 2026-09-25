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

## Directory Structure

```
Chatbot/
├── app.py                      # Flask REST API backend
├── train.py                    # Model training pipeline
├── render.yaml                 # Render deployment configuration
├── Procfile                    # Web process command for cloud PaaS
├── requirements.txt            # Python dependencies
├── data/
│   └── intents.json            # Conversational intents dataset
├── model/
│   ├── chatbot_model.keras     # Trained Deep Learning model
│   ├── words.json              # Vocabulary tokens
│   └── classes.json            # Intent classification tags
├── src/
│   ├── __init__.py
│   ├── chatbot.py              # Inference engine
│   ├── model.py                # Keras Sequential architecture definition
│   ├── nlp_utils.py            # Tokenizer, Porter stemmer, Bag-of-Words
│   └── speech_service.py       # Speech recognition service
├── static/
│   ├── css/style.css           # Glassmorphic responsive styling
│   └── js/app.js               # Client speech recognition & API caller
├── templates/
│   └── index.html              # Frontend user interface
├── test/                       # Modular test suite
│   ├── __init__.py
│   ├── test_helpers.py         # Test fixture generators
│   ├── test_model.py           # NLP & model tests
│   ├── test_speech.py          # Speech recognition tests
│   ├── test_api.py             # Flask API integration tests
│   └── run_all_tests.py        # Master test runner
├── planning/
│   ├── project_plan.md         # Phased implementation plan
│   ├── research_and_architecture.md # Technical research and design
│   └── memory.md               # Append-only project progress log
└── report.md                   # Technical evaluation report
```

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

---

## Deployment to Render / Railway

### Render Deployment Steps:
1. Push this repository to GitHub.
2. Log in to [Render](https://render.com) and create a **New Web Service**.
3. Connect your GitHub repository.
4. Set the following settings:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt && python train.py`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`
   - **Health Check Path:** `/health`
5. Click **Create Web Service**. Your app will be live within 2-3 minutes.

### Railway Deployment Steps:
1. Create a **New Project** on [Railway](https://railway.app).
2. Select **Deploy from GitHub repo**.
3. Railway automatically detects `Procfile` and `requirements.txt`.
4. Add environment variable `PORT=5000` if prompted.
