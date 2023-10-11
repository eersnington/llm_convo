from abc import ABC, abstractmethod
from typing import Optional
import os
import tempfile
import subprocess

import pyaudio
import wave
import logging


class TTSClient(ABC):
    @abstractmethod
    def text_to_mp3(self, text: str) -> str:
        pass


class WhisperTTS(TTSClient):
    def text_to_mp3(self, text: str) -> str:
        logging.critical(f"Generating TTS for {text}")
        if text.__contains__("<Hangup/>"):
            logging.info("Hangup detected, removing from response")
            text = text.replace("<Hangup/>", "")
            return f"<Say>{text}</Say><Hangup/>"
        else:
            return f"<Say>{text}</Say>"
