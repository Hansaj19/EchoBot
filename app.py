"""
Flask Web Application for Voice-Enabled Chatbot.
Exposes RESTful API endpoints for text chat and speech processing,
and serves the responsive frontend interface.
"""

import os
import io
from flask import Flask, render_template, request, jsonify
from src.chatbot import get_chatbot
from src.speech_service import get_speech_service

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB limit
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


@app.route("/")
def index():
    """Serves the main Voice-Enabled Chatbot interactive UI."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for deployment monitoring (Render/Railway)."""
    try:
        bot = get_chatbot()
        model_loaded = bot.model is not None
    except Exception:
        model_loaded = False

    return jsonify({
        "status": "healthy" if model_loaded else "degraded",
        "service": "voice-enabled-chatbot",
        "model_loaded": model_loaded
    }), (200 if model_loaded else 500)


@app.route("/api/intents", methods=["GET"])
def get_intents():
    """Returns list of supported conversational intents."""
    try:
        bot = get_chatbot()
        intents_list = list(bot.intents.keys())
        return jsonify({
            "success": True,
            "intents": intents_list,
            "total": len(intents_list)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Processes text message query via Deep Learning intent classifier.
    Request JSON: {"message": "Hello"}
    """
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "Invalid request. 'message' field is required."}), 400

    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    try:
        bot = get_chatbot()
        result = bot.get_response(user_message)
        return jsonify({
            "success": True,
            "query": result["query"],
            "intent": result["intent"],
            "confidence": result["confidence"],
            "response": result["response"],
            "source": "text"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/voice", methods=["POST"])
def voice():
    """
    Processes audio voice input:
    1. Transcribes speech to text via SpeechRecognition.
    2. Feeds recognized text to the Deep Learning intent classifier.
    3. Returns transcription, detected intent, confidence, and bot response.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided in 'audio' field."}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "Empty audio file."}), 400

    try:
        speech_service = get_speech_service()
        audio_bytes = audio_file.read()

        stt_result = speech_service.transcribe_audio_file(audio_bytes)

        if not stt_result["success"] or not stt_result["text"]:
            return jsonify({
                "success": False,
                "error": stt_result.get("error", "Could not transcribe audio."),
                "query": "",
                "response": "I couldn't hear or transcribe your speech clearly. Please try speaking again or type your message."
            }), 200

        recognized_text = stt_result["text"]
        bot = get_chatbot()
        chat_result = bot.get_response(recognized_text)

        return jsonify({
            "success": True,
            "query": recognized_text,
            "intent": chat_result["intent"],
            "confidence": chat_result["confidence"],
            "response": chat_result["response"],
            "source": "voice"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
