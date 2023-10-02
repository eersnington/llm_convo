import io
import os
import tempfile
import queue
import functools
import logging

from pydub import AudioSegment
import speech_recognition as sr
from faster_whisper import WhisperModel


@functools.cache
def get_whisper_model(size: str = "small.en"):
    logging.info(f"Loading whisper {size}")
    STTmodel = WhisperModel(size, device="cuda", compute_type="int8_float16")
    return STTmodel


def asr_transcript(input, model):
    segments, info = model.transcribe(input, beam_size=5)

    res = []
    for segment in segments:
        res.append(segment.text)

    return " ".join(res)


class WhisperMicrophone:
    def __init__(self):
        self.audio_model = get_whisper_model()
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 500
        self.recognizer.pause_threshold = 0.8
        self.recognizer.dynamic_energy_threshold = False

    def get_transcription(self) -> str:
        with sr.Microphone(sample_rate=16000) as source:
            logging.info("Waiting for mic...")
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = os.path.join(tmp, "mic.wav")
                audio = self.recognizer.listen(source)
                data = io.BytesIO(audio.get_wav_data())
                audio_clip = AudioSegment.from_file(data)
                audio_clip.export(tmp_path, format="wav")
                result = asr_transcript(tmp_path, self.audio_model)
            predicted_text = result
        return predicted_text


class _TwilioSource(sr.AudioSource):
    def __init__(self, stream):
        self.stream = stream
        self.CHUNK = 1024
        self.SAMPLE_RATE = 8000
        self.SAMPLE_WIDTH = 2

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class _QueueStream:
    def __init__(self):
        self.q = queue.Queue(maxsize=-1)

    def read(self, chunk: int) -> bytes:
        return self.q.get()

    def write(self, chunk: bytes):
        self.q.put(chunk)


class WhisperTwilioStream:
    def __init__(self):
        self.audio_model = get_whisper_model()
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.pause_threshold = 2.5
        self.recognizer.dynamic_energy_threshold = False
        self.stream = None

    def get_transcription(self) -> str:
        self.stream = _QueueStream()
        with _TwilioSource(self.stream) as source:
            logging.info("Waiting for twilio caller...")
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = os.path.join(tmp, "mic.wav")
                audio = self.recognizer.listen(source)
                data = io.BytesIO(audio.get_wav_data())
                audio_clip = AudioSegment.from_file(data)
                audio_clip.export(tmp_path, format="wav")
                result = asr_transcript(tmp_path, self.audio_model)
        predicted_text = result
        self.stream = None
        return predicted_text
