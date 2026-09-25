"""
Speech Recognition Service.
Handles server-side audio file transcription using SpeechRecognition.
Provides fallback transcription capabilities for clients where browser-native
Web Speech API might be unavailable.
"""

import os
import io
import speech_recognition as sr


class SpeechService:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        # Adjust recognizer thresholds for better speech capture
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True

    def transcribe_audio_file(self, file_path_or_bytes) -> dict:
        """
        Transcribes speech from an audio file path or audio byte stream.
        Returns a dict:
        - success: bool
        - text: transcribed string
        - error: error message if unsuccessful
        """
        try:
            if isinstance(file_path_or_bytes, (str, os.PathLike)):
                with sr.AudioFile(file_path_or_bytes) as source:
                    audio_data = self.recognizer.record(source)
            elif isinstance(file_path_or_bytes, (bytes, bytearray)):
                byte_io = io.BytesIO(file_path_or_bytes)
                with sr.AudioFile(byte_io) as source:
                    audio_data = self.recognizer.record(source)
            else:
                # File-like object
                with sr.AudioFile(file_path_or_bytes) as source:
                    audio_data = self.recognizer.record(source)

            # Transcribe via Google Speech Recognition API
            text = self.recognizer.recognize_google(audio_data)
            return {
                "success": True,
                "text": text,
                "error": None
            }
        except sr.UnknownValueError:
            return {
                "success": False,
                "text": "",
                "error": "Speech was unintelligible or no audio detected."
            }
        except sr.RequestError as e:
            return {
                "success": False,
                "text": "",
                "error": f"Speech recognition service error: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "text": "",
                "error": f"Audio processing error: {str(e)}"
            }


_speech_service_instance = None


def get_speech_service() -> SpeechService:
    global _speech_service_instance
    if _speech_service_instance is None:
        _speech_service_instance = SpeechService()
    return _speech_service_instance
