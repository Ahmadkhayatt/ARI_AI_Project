import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google-tts-key.json"  # Google TTS auth

import tempfile
import time
import json
import requests
import threading
import wave
import numpy as np
import pyaudio
import webrtcvad
from collections import deque
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


# === Record Until Silence ===
def record_until_silence(filename="record.wav", aggressiveness=2, timeout=3, rate=16000, chunk_duration_ms=30):
    vad = webrtcvad.Vad(aggressiveness)
    chunk_size = int(rate * chunk_duration_ms / 1000)
    audio_format = pyaudio.paInt16
    channels = 1

    p = pyaudio.PyAudio()
    stream = p.open(format=audio_format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk_size)
    print("🎙️ Recording...")

    frames = []
    silence_counter = 0
    triggered = False

    try:
        while True:
            frame = stream.read(chunk_size, exception_on_overflow=False)
            is_speech = vad.is_speech(frame, rate)

            if is_speech:
                if not triggered:
                    triggered = True
                    print("🔊 Voice detected.")
                frames.append(frame)
                silence_counter = 0
            elif triggered:
                frames.append(frame)
                silence_counter += 1
                if silence_counter > int(timeout * 1000 / chunk_duration_ms):
                    print("🤫 Silence detected, stopping.")
                    break
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(audio_format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

    return filename

# === Transcribe Audio with Google STT ===
def transcribe_google_stt(file_path, language="tr-TR"):
    client = speech.SpeechClient()
    with open(file_path, 'rb') as f:
        audio_content = f.read()

    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language,
    )

    response = client.recognize(config=config, audio=audio)
    return " ".join([result.alternatives[0].transcript for result in response.results])

# === Originate Call through ARI ===
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


# === Multi-Turn AI Interaction ===
def interact_with_user(channel_id):
    stop_words = ["exit", "kapat", "bitir"]
    closing_phrases = ["teşekkür ederiz", "anketimiz sona erdi", "katılımınız için teşekkür ederiz"]

    while True:
        audio_path = "/tmp/user_recording.wav"
        record_until_silence(audio_path)

        try:
            user_text = transcribe_google_stt(audio_path)
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
                time.sleep(5)
            break

        response = chat_session.send_message(user_text)
        reply = response.text.strip()
        print("🤖 Gemini:", reply)

        media_id = speak_to_file(reply)
        if media_id:
            requests.post(
                f"{BASE_URL}/channels/{channel_id}/play",
                params={"media": f"sound:{media_id}"},
                auth=HTTPBasicAuth(ARI_USER, ARI_PASSWORD)
            )
            time.sleep(5)

        if any(phrase in reply.lower() for phrase in closing_phrases):
            print("📴 Ending call based on Gemini response.")
            break

    # End the call
    requests.delete(f"{BASE_URL}/channels/{channel_id}", auth=HTTPBasicAuth(ARI_USER, ARI_PASSWORD))


# === Handle ARI Events ===
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
            # Run user interaction in a separate thread
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

# === Main Entry Point ===  
if __name__ == "__main__":
    start_websocket()
