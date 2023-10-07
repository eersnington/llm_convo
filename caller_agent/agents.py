from typing import List, Optional
from abc import ABC, abstractmethod
from vllm import LLM

from caller_agent.llama_agent import LlamaAgent
from caller_agent.twilio_io import TwilioCallSession
from caller_agent.audio_output import TTSClient, GoogleTTS


class ChatAgent(ABC):
    @abstractmethod
    def get_response(self, transcript: List[str]) -> str:
        pass

    def start(self):
        pass


class TerminalInPrintOut(ChatAgent):
    def get_response(self, transcript: List[str]) -> str:
        return input(" response > ")


class LlamaChatAgent(ChatAgent):
    def __init__(self, init_phrase: Optional[str] = None):
        self.llama_chat = LlamaAgent(init_phrase=init_phrase)
        self.init_phrase = init_phrase

    def get_response(self, transcript: List[str]) -> str:
        if len(transcript) > 0:
            response = self.llama_chat.get_response(transcript[-1])
        else:
            response = self.init_phrase
        return response


class TwilioCaller(ChatAgent):
    def __init__(self, session: TwilioCallSession, tts: Optional[TTSClient] = None, thinking_phrase: str = "Okay"):
        self.session = session
        self.speaker = tts or GoogleTTS()
        self.thinking_phrase = thinking_phrase

    def _say(self, text: str):
        self.session.play(self.speaker.text_to_mp3(text))

    def get_response(self, transcript: List[str]) -> str:
        if len(transcript) > 0:
            self._say(transcript[-1])
        resp = self.session.sst_stream.get_transcription()
        return resp
