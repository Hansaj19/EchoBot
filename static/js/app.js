/**
 * EchoBot Frontend Client Application.
 * Apple iMessage White & Blue Theme.
 * Handles Speech Recognition (STT), Deep Learning Chatbot API communication,
 * Text-to-Speech (TTS) audio playback, and Apple iMessage UI rendering.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const chatMessages = document.getElementById("chat-messages");
  const chatForm = document.getElementById("chat-form");
  const userInput = document.getElementById("user-input");
  const micBtn = document.getElementById("mic-btn");
  const voiceOverlay = document.getElementById("voice-overlay");
  const liveSpeechPreview = document.getElementById("live-speech-preview");
  const stopListeningBtn = document.getElementById("stop-listening-btn");
  const statusBar = document.getElementById("status-bar");
  const statusText = document.getElementById("status-text");
  const ttsToggleBtn = document.getElementById("tts-toggle-btn");
  const clearChatBtn = document.getElementById("clear-chat-btn");
  const quickChips = document.querySelectorAll(".chip");

  // State
  let isListening = false;
  let ttsEnabled = true;
  let recognition = null;
  let mediaRecorder = null;
  let audioChunks = [];

  // Check Web Speech API Availability
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onstart = () => {
      isListening = true;
      micBtn.classList.add("recording");
      voiceOverlay.classList.remove("hidden");
      statusBar.className = "imessage-status-bar listening";
      statusText.textContent = "Listening... Speak clearly into microphone";
      liveSpeechPreview.textContent = "Listening... Speak now";
    };

    recognition.onresult = (event) => {
      let interimTranscript = "";
      let finalTranscript = "";

      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript;
        } else {
          interimTranscript += event.results[i][0].transcript;
        }
      }

      const displayTranscript = finalTranscript || interimTranscript;
      if (displayTranscript) {
        liveSpeechPreview.textContent = `"${displayTranscript}"`;
      }

      if (finalTranscript.trim()) {
        recognition.stop();
        handleUserMessage(finalTranscript.trim(), true);
      }
    };

    recognition.onerror = (event) => {
      console.warn("Speech recognition error:", event.error);
      stopListening();
      if (event.error === "not-allowed") {
        updateStatus("Microphone access denied. Please grant permissions.", "error");
      } else if (event.error === "no-speech") {
        updateStatus("No speech detected. Tap microphone to speak again.", "ready");
      } else {
        updateStatus(`Speech error (${event.error}). You can also type.`, "ready");
      }
    };

    recognition.onend = () => {
      stopListening();
    };
  } else {
    console.info("Web Speech API not natively supported; falling back to MediaRecorder audio upload.");
  }

  function startListening() {
    if (isListening) {
      stopListening();
      return;
    }

    if (recognition) {
      try {
        recognition.start();
      } catch (err) {
        console.error("Failed to start speech recognition:", err);
      }
    } else {
      // Fallback: Use MediaStream / MediaRecorder to record audio file
      startAudioRecordingFallback();
    }
  }

  function stopListening() {
    isListening = false;
    micBtn.classList.remove("recording");
    voiceOverlay.classList.add("hidden");
    statusBar.className = "imessage-status-bar";
    statusText.textContent = "iMessage with EchoBot AI";

    if (recognition) {
      try {
        recognition.stop();
      } catch (e) {}
    }

    if (mediaRecorder && mediaRecorder.state === "recording") {
      mediaRecorder.stop();
    }
  }

  async function startAudioRecordingFallback() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunks = [];
      mediaRecorder = new MediaRecorder(stream);

      mediaRecorder.onstart = () => {
        isListening = true;
        micBtn.classList.add("recording");
        voiceOverlay.classList.remove("hidden");
        statusBar.className = "imessage-status-bar listening";
        statusText.textContent = "Recording audio (Fallback Mode)...";
        liveSpeechPreview.textContent = "Recording... Tap Done when finished.";
      };

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunks.push(e.data);
        }
      };

      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
        await sendAudioToBackend(audioBlob);
      };

      mediaRecorder.start();
    } catch (err) {
      console.error("Microphone access failed:", err);
      updateStatus("Microphone access failed. Please type your query.", "error");
    }
  }

  async function sendAudioToBackend(audioBlob) {
    updateStatus("Transcribing and processing audio...", "processing");
    const formData = new FormData();
    formData.append("audio", audioBlob, "user_speech.wav");

    try {
      const response = await fetch("/api/voice", {
        method: "POST",
        body: formData
      });
      const data = await response.json();

      if (data.success && data.query) {
        renderUserMessage(data.query, true);
        renderBotMessage(data.response, data.intent, data.confidence);
        if (ttsEnabled) {
          speakText(data.response);
        }
        updateStatus("Delivered", "ready");
      } else {
        renderBotMessage(data.response || "Could not recognize audio. Please try again.", "fallback", 0.0);
        updateStatus("Could not transcribe speech. Try again.", "ready");
      }
    } catch (err) {
      console.error("Voice API error:", err);
      renderBotMessage("Error communicating with server. Please try again.", "error", 0.0);
      updateStatus("Connection error.", "error");
    }
  }

  // Handle Text Submission
  async function handleUserMessage(message, isVoice = false) {
    if (!message || !message.trim()) return;

    renderUserMessage(message, isVoice);
    userInput.value = "";
    updateStatus("EchoBot is typing...", "processing");

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: message })
      });
      const data = await response.json();

      if (data.success) {
        renderBotMessage(data.response, data.intent, data.confidence);
        if (ttsEnabled) {
          speakText(data.response);
        }
        updateStatus("iMessage with EchoBot AI", "ready");
      } else {
        renderBotMessage(data.error || "An error occurred.", "fallback", 0.0);
        updateStatus("Error processing message.", "error");
      }
    } catch (err) {
      console.error("Chat error:", err);
      renderBotMessage("Server unreachable. Please make sure the service is online.", "error", 0.0);
      updateStatus("Server error.", "error");
    }
  }

  // Render User Message (Apple iMessage Blue Outgoing Bubble)
  function renderUserMessage(text, isVoice = false) {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message user-message";

    const badgeHtml = isVoice
      ? `<span class="badge badge-voice">Voice</span>`
      : "";

    msgDiv.innerHTML = `
      <div class="message-content">
        <div class="message-text">${escapeHtml(text)}</div>
        <div class="message-meta-footer">
          ${badgeHtml}
          <span class="delivered-tag">Delivered</span>
        </div>
      </div>
    `;

    chatMessages.appendChild(msgDiv);
    scrollToBottom();
  }

  // Render Bot Message (Apple iMessage Gray Incoming Bubble)
  function renderBotMessage(text, intent, confidence) {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message bot-message";

    const confPercent = confidence ? Math.round(confidence * 100) : 0;
    const intentBadge = intent ? `<span class="badge badge-intent">${escapeHtml(intent)}</span>` : "";
    const confBadge = confPercent > 0 ? `<span class="badge badge-confidence">${confPercent}%</span>` : "";

    msgDiv.innerHTML = `
      <div class="message-avatar">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="3" y="11" width="18" height="10" rx="2"></rect>
          <circle cx="12" cy="5" r="2"></circle>
          <path d="M12 7v4"></path>
          <line x1="8" y1="16" x2="8.01" y2="16"></line>
          <line x1="16" y1="16" x2="16.01" y2="16"></line>
        </svg>
      </div>
      <div class="message-content">
        <div class="message-sender-label">EchoBot AI</div>
        <div class="message-text">${escapeHtml(text)}</div>
        <div class="message-meta-footer">
          ${intentBadge}
          ${confBadge}
          <button class="speak-msg-btn" title="Listen to this response">
            <svg viewBox="0 0 24 24" fill="currentColor">
              <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
              <path d="M15.54 8.46a5 5 0 0 1 0 7.07" fill="none" stroke="currentColor" stroke-width="2"></path>
            </svg>
            <span>Play</span>
          </button>
        </div>
      </div>
    `;

    // Attach individual TTS playback
    const speakBtn = msgDiv.querySelector(".speak-msg-btn");
    speakBtn.addEventListener("click", () => {
      speakText(text);
    });

    chatMessages.appendChild(msgDiv);
    scrollToBottom();
  }

  // Text to Speech
  function speakText(text) {
    if (!("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel(); // Stop any pending speech

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.lang = "en-US";

    utterance.onstart = () => {
      updateStatus("EchoBot speaking...", "processing");
    };

    utterance.onend = () => {
      updateStatus("iMessage with EchoBot AI", "ready");
    };

    utterance.onerror = () => {
      updateStatus("iMessage with EchoBot AI", "ready");
    };

    window.speechSynthesis.speak(utterance);
  }

  // Helper Functions
  function updateStatus(message, state = "ready") {
    statusText.textContent = message;
    if (state === "listening") {
      statusBar.className = "imessage-status-bar listening";
    } else if (state === "processing") {
      statusBar.className = "imessage-status-bar processing";
    } else {
      statusBar.className = "imessage-status-bar";
    }
  }

  function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function escapeHtml(string) {
    const div = document.createElement("div");
    div.textContent = string;
    return div.innerHTML;
  }

  // Event Listeners
  micBtn.addEventListener("click", startListening);
  stopListeningBtn.addEventListener("click", stopListening);

  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const query = userInput.value.trim();
    if (query) {
      handleUserMessage(query, false);
    }
  });

  // Quick Chips Click
  quickChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const query = chip.getAttribute("data-query");
      if (query) {
        handleUserMessage(query, false);
      }
    });
  });

  // TTS Toggle
  ttsToggleBtn.addEventListener("click", () => {
    ttsEnabled = !ttsEnabled;
    const soundOn = ttsToggleBtn.querySelector(".icon-sound-on");
    const soundOff = ttsToggleBtn.querySelector(".icon-sound-off");

    if (ttsEnabled) {
      ttsToggleBtn.classList.add("active");
      soundOn.style.display = "block";
      soundOff.style.display = "none";
      ttsToggleBtn.setAttribute("title", "Voice Output: ON");
    } else {
      ttsToggleBtn.classList.remove("active");
      soundOn.style.display = "none";
      soundOff.style.display = "block";
      ttsToggleBtn.setAttribute("title", "Voice Output: OFF");
      if ("speechSynthesis" in window) {
        window.speechSynthesis.cancel();
      }
    }
  });

  // Clear Chat (Resets to Apple iMessage Welcome Message)
  clearChatBtn.addEventListener("click", () => {
    chatMessages.innerHTML = `
      <div class="date-divider">
        <span>Today</span>
      </div>
      <div class="message bot-message">
        <div class="message-avatar">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="3" y="11" width="18" height="10" rx="2"></rect>
            <circle cx="12" cy="5" r="2"></circle>
            <path d="M12 7v4"></path>
            <line x1="8" y1="16" x2="8.01" y2="16"></line>
            <line x1="16" y1="16" x2="16.01" y2="16"></line>
          </svg>
        </div>
        <div class="message-content">
          <div class="message-sender-label">EchoBot AI</div>
          <div class="message-text">Chat cleared! Tap the microphone button or type an iMessage to begin a new conversation.</div>
          <div class="message-meta-footer">
            <span class="badge badge-system">Conversation Reset</span>
          </div>
        </div>
      </div>
    `;
    updateStatus("iMessage with EchoBot AI", "ready");
  });
});
