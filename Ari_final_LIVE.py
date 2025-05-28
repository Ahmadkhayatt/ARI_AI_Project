import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google-tts-key.json"

import tempfile
import time
import json
import requests
import threading
import wave
import queue
import numpy as np
import pyaudio
import webrtcvad
from dotenv import load_dotenv
from google.cloud import texttospeech, speech
import google.generativeai as genai
from requests.auth import HTTPBasicAuth
import websocket

# === Load environment variables ===
load_dotenv()

# === Configs ===
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ARI_USER = 'ai'
ARI_PASSWORD = 'ai_secret'
ARI_HOST = 'localhost'
ARI_PORT = 8088
ARI_APP = 'aiagent'
BASE_URL = f'http://{ARI_HOST}:{ARI_PORT}/ari'

# === Gemini Setup ===
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    },
)
with open('PromptNew2.txt', "r", encoding="utf-8") as file:
    prompt = file.read()
chat_session = model.start_chat(history=[{"role": "user", "parts": [prompt]}])


# === Google TTS ===
def speak_to_file(text, lang="tr-TR"):
    try:
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=lang,
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

        mp3_path = os.path.join(tempfile.gettempdir(), "tts.mp3")
        wav_path = "/var/lib/asterisk/sounds/ai.wav"
        with open(mp3_path, "wb") as out:
            out.write(response.audio_content)
        os.system(f"ffmpeg -y -i {mp3_path} -ar 8000 -ac 1 -f wav {wav_path}")
        return "ai"
    except Exception as e:
        print("Google TTS Error:", e)
        return None


# === VAD-Based Playback Monitor ===
class InterruptPlaybackMonitor(threading.Thread):
    def __init__(self, channel_id, vad_aggressiveness=2, rate=16000, duration=2):
        super().__init__()
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.rate = rate
        self.chunk_duration_ms = 30
        self.chunk_size = int(rate * self.chunk_duration_ms / 1000)
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.duration = duration
        self.threshold_chunks = int((1000 / self.chunk_duration_ms) * duration)
        self._stop_event = threading.Event()
        self.channel_id = channel_id

    def run(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=self.audio_format, channels=self.channels, rate=self.rate, input=True,
                        frames_per_buffer=self.chunk_size)

        print("🎧 Monitoring for interruption...")
        try:
            speech_chunks = 0
            while not self._stop_event.is_set():
                frame = stream.read(self.chunk_size, exception_on_overflow=False)
                if self.vad.is_speech(frame, self.rate):
                    print("🎤 VOICE!")
                    speech_chunks += 1
                    if speech_chunks >= self.threshold_chunks:
                        print("🛑 Voice detected — stopping playback.")
                        self.stop_playback()
                        break
                else:
                    speech_chunks = 0
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    def stop_playback(self):
        
        
        print("🛑 Sending stop playback command to ARI...")
        requests.delete(f"{BASE_URL}/channels/{self.channel_id}/play", auth=HTTPBasicAuth(ARI_USER, ARI_PASSWORD))

    def stop(self):
        self._stop_event.set()


# === Google Streaming STT ===
class MicrophoneStream:
    def __init__(self, rate=16000, chunk_size=1024):
        self.rate = rate
        self.chunk_size = chunk_size
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self.audio_interface = pyaudio.PyAudio()
        self.stream = self.audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self.stream.stop_stream()
        self.stream.close()
        self.closed = True
        self._buff.put(None)
        self.audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)


def listen_google_streaming(language="tr-TR", silence_timeout=3):
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language,
        max_alternatives=1,
        enable_automatic_punctuation=True,
    )
    streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=False)

    with MicrophoneStream() as stream:
        audio_generator = stream.generator()
        requests_gen = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator)

        print("🎤 Listening via Google STT...")
        responses = client.streaming_recognize(config=streaming_config, requests=requests_gen)

        for response in responses:
            if not response.results:
                continue
            result = response.results[0]
            if result.is_final:
                transcript = result.alternatives[0].transcript.strip()
                print("🗣️ Final Transcript:", transcript)
                return transcript


# === Main Multi-Turn AI Conversation ===
def interact_with_user(channel_id):
    stop_words = ["exit", "kapat", "bitir"]
    closing_phrases = ["teşekkür ederiz", "anketimiz sona erdi", "katılımınız için teşekkür ederiz"]

    while True:
        # Listen via Google Streaming STT
        try:
            user_text = listen_google_streaming()
        except Exception as e:
            print("❌ STT error:", e)
            user_text = ""

        if not user_text:
            print("⚠️ No input detected, skipping.")
            continue

        print("📝 User said:", user_text)
        if any(word in user_text.lower() for word in stop_words):
            farewell = "Anket sonlandırıldı. Katılımınız için teşekkür ederiz."
            media_id = speak_to_file(farewell)
            if media_id:
                requests.post(
                    f"{BASE_URL}/channels/{channel_id}/play",
                    params={"media": f"sound:{media_id}"},
                    auth=HTTPBasicAuth(ARI_USER, ARI_PASSWORD)
                )
                time.sleep(6)
            break

        response = chat_session.send_message(user_text)
        reply = response.text.strip()
        print("🤖 Gemini:", reply)

        # Speak and allow interruption
        media_id = speak_to_file(reply)
        if media_id:
            interrupt_monitor = InterruptPlaybackMonitor(channel_id)
            interrupt_monitor.start()

            requests.post(
                f"{BASE_URL}/channels/{channel_id}/play",
                params={"media": f"sound:{media_id}"},
                auth=HTTPBasicAuth(ARI_USER, ARI_PASSWORD)
            )

            # Wait a moment for playback to start
            time.sleep(1)

            # Wait for interruption or 6 seconds (typical sentence length)
            interrupt_monitor.join(timeout=0.4)
            interrupt_monitor.stop()

        if any(phrase in reply.lower() for phrase in closing_phrases):
            print("📴 Ending call based on Gemini response.")
            break

    # End the call
    requests.delete(f"{BASE_URL}/channels/{channel_id}", auth=HTTPBasicAuth(ARI_USER, ARI_PASSWORD))


# === Outbound Call Originate ===
def originate_call():
    url = f'{BASE_URL}/channels'
    data = {
        "endpoint": "PJSIP/7001",
        "extension": "3001",
        "context": "ai-survey",
        "priority": "1",
        "app": ARI_APP,
        "callerId": "AI Bot"
    }
    response = requests.post(url, data=data, auth=HTTPBasicAuth(ARI_USER, ARI_PASSWORD))
    if response.status_code == 200:
        print("📞 Outbound call originated to 7001")
    else:
        print("❌ Call failed:", response.status_code, response.text)


# === WebSocket Event Handlers ===
def on_message(ws, message):
    event = json.loads(message)
    print("📡 Event:", event['type'])

    if event['type'] == 'StasisStart':
        channel_id = event['channel']['id']
        greeting = "Merhaba, sizinle kısa bir anket yapmak istiyoruz."
        media_id = speak_to_file(greeting)
        if media_id:
            requests.post(
                f"{BASE_URL}/channels/{channel_id}/play",
                params={"media": f"sound:{media_id}"},
                auth=HTTPBasicAuth(ARI_USER, ARI_PASSWORD)
            )
            # Start interaction loop in new thread
            threading.Thread(target=interact_with_user, args=(channel_id,)).start()

    elif event['type'] == 'StasisEnd':
        print("📞 Call ended and channel destroyed.")


def on_open(ws):
    print("✅ WebSocket connected to ARI")
    originate_call()

def on_error(ws, error):
    print("❗ WebSocket Error:", error)

def on_close(ws, *args):
    print("🔌 WebSocket closed")


def start_websocket():
    ws_url = f'ws://{ARI_HOST}:{ARI_PORT}/ari/events?app={ARI_APP}&api_key={ARI_USER}:{ARI_PASSWORD}'
    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()


# === App Entry ===
if __name__ == "__main__":
    start_websocket()
