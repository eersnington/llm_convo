import threading
import logging
import os
import base64
import json
import time

from gevent.pywsgi import WSGIServer
from twilio.rest import Client
from flask import Flask, send_from_directory
from flask_sock import Sock
import simple_websocket
import audioop

from caller_agent.audio_input import WhisperTwilioStream


XML_MEDIA_STREAM = """
<Response>
    <Start>
        <Stream name="Audio Stream" url="wss://{host}/audiostream" />
    </Start>
    <Pause length="60"/>
</Response>
"""


class TwilioServer:
    def __init__(self, remote_host: str, port: int, static_dir: str):
        self.app = Flask(__name__)
        self.sock = Sock(self.app)
        self.remote_host = remote_host
        self.port = port
        self.static_dir = static_dir
        self.server_thread = threading.Thread(target=self._start)
        self.on_session = None

        account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        auth_token = os.environ["TWILIO_AUTH_TOKEN"]
        self.client = Client(account_sid, auth_token)
        self.from_phone = self.client.incoming_phone_numbers.list()[0].phone_number

        self.client.incoming_phone_numbers.list()[0].update(voice_url="https://"+remote_host+"/incoming-voice")

        @self.app.route("/incoming-voice", methods=["POST"])
        def incoming_voice():
            return XML_MEDIA_STREAM.format(host=self.remote_host), 200, {'Content-Type': 'text/xml'}

        @self.sock.route("/audiostream", websocket=True)
        def on_media_stream(ws):
            session = TwilioCallSession(ws, self.client, remote_host=self.remote_host, static_dir=self.static_dir)
            if self.on_session is not None:
                thread = threading.Thread(target=self.on_session, args=(session,))
                thread.start()
            session.start_session()

    def start_call(self, to_phone: str):
        self.client.calls.create(
            twiml=XML_MEDIA_STREAM.format(host=self.remote_host),
            to=to_phone,
            from_=self.from_phone,
        )

    def _start(self):
        logging.info("Starting Twilio Server")
        WSGIServer(("0.0.0.0", self.port), self.app).serve_forever()

    def start(self):
        self.server_thread.start()


class TwilioCallSession:
    def __init__(self, ws, client: Client, remote_host: str, static_dir: str):
        self.ws = ws
        self.client = client
        self.sst_stream = WhisperTwilioStream()
        self.remote_host = remote_host
        self.static_dir = static_dir
        self._call = None

    def media_stream_connected(self):
        return self._call is not None

    def _read_ws(self):
        while True:
            try:
                message = self.ws.receive()
            except simple_websocket.ws.ConnectionClosed:
                logging.warn("Call media stream connection lost.")
                break
            if message is None:
                logging.warn("Call media stream closed.")
                break

            data = json.loads(message)
            if data["event"] == "start":
                logging.info("Call connected, " + str(data["start"]))
                self._call = self.client.calls(data["start"]["callSid"])
            elif data["event"] == "media":
                media = data["media"]
                chunk = base64.b64decode(media["payload"])
                if self.sst_stream.stream is not None:
                    self.sst_stream.stream.write(audioop.ulaw2lin(chunk, 2))
            elif data["event"] == "stop":
                logging.info("Call media stream ended.")
                break

    def play(self, audio: str):
        self._call.update(
            twiml=f'<Response><Say>{audio}</Say></Response>'
        )

    def start_session(self):
        self._read_ws()