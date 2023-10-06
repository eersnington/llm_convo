from gevent import monkey

monkey.patch_all()

from caller_agent.agents import LlamaChatAgent, TwilioCaller
from caller_agent.twilio_io import TwilioServer
from caller_agent.conversation import run_conversation
from caller_agent.llama_agent import get_llm_model
from caller_agent.audio_input import get_whisper_model

from pyngrok import ngrok
from vllm import LLM

import logging
import time
import os
import tempfile

logging.getLogger().setLevel(logging.INFO)

port=8081
static_dir = os.path.join(tempfile.gettempdir(), "twilio_static")
os.makedirs(static_dir, exist_ok=True)
ngrok_http = ngrok.connect(port)
remote_host = ngrok_http.public_url.split("//")[1]

logging.info(f"Starting server at {remote_host} from local:{port}, serving static content from {static_dir}")
logging.info(f"Set call webhook to https://{remote_host}/incoming-voice")

get_llm_model()
get_whisper_model()

tws = TwilioServer(remote_host=remote_host, port=port, static_dir=static_dir)

tws.start()
agent_a = LlamaChatAgent(init_phrase="Hi there, I'm Ruby. How can I help you today?")

def run_chat(sess):
    agent_b = TwilioCaller(sess)
    while not agent_b.session.media_stream_connected():
        time.sleep(0.1)
    run_conversation(agent_a, agent_b)

tws.on_session = run_chat

# Outbound call
# tws.start_call("+919952062221")