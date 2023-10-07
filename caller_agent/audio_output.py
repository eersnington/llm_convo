from abc import ABC, abstractmethod
from typing import Optional
import os
import tempfile
import subprocess

from gtts import gTTS
import pyaudio
import wave


class TTSClient(ABC):
    @abstractmethod
    def text_to_mp3(self, text: str) -> str:
        pass

    def play_text(self, text: str) -> str:
        tmp_mp3 = self.text_to_mp3(text)
        tmp_wav = tmp_mp3.replace(".mp3", ".wav")
        subprocess.call(["ffmpeg", "-hide_banner", "-loglevel",
                        "error", "-y", "-i", tmp_mp3, tmp_wav])

        wf = wave.open(tmp_wav, "rb")
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=audio.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
        )

        data = wf.readframes(1024)
        while data != b"":
            stream.write(data)
            data = wf.readframes(1024)

        stream.close()
        audio.terminate()


class GoogleTTS(TTSClient):
    def text_to_mp3(self, text: str) -> str:
        if text.__contains__("<Hangup/>"):
            text = text.replace("<Hangup/>", "")
            return f"<Say>{text}</Say><Hangup/>"
        return f"<Say>{text}</Say>"
