from gevent import monkey

monkey.patch_all()

from caller_agent.agents import LlamaChatAgent, TwilioCaller
from caller_agent.twilio_io import TwilioServer
from caller_agent.conversation import run_conversation

from pyngrok import ngrok
from vllm import LLM

import logging
import time
import os
import tempfile

logging.getLogger().setLevel(logging.INFO)

port=8080
static_dir = os.path.join(tempfile.gettempdir(), "twilio_static")
os.makedirs(static_dir, exist_ok=True)
ngrok_http = ngrok.connect(8080)
remote_host = ngrok_http.public_url.split("//")[1]

chat_llm = LLM(model="TheBloke/Llama-2-7B-chat-AWQ", quantization="awq")

tws = TwilioServer(remote_host=remote_host, port=port, static_dir=static_dir)

tws.start()
agent_a = LlamaChatAgent(
    llm=chat_llm,
    init_phrase="Hello?",
)

def run_chat(sess):
    agent_b = TwilioCaller(sess)
    while not agent_b.session.media_stream_connected():
        time.sleep(0.1)
    run_conversation(agent_a, agent_b)

tws.on_session = run_chat

# Outbound call
# tws.start_call("+18321231234")