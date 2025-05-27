import os
import pygame
import tempfile
import time
import requests
import json
import pyaudio
import wave
import whisper
import webrtcvad
import numpy as np
import collections
import torch
import warnings
# from google import genai
import google.generativeai as genai
import os
from dotenv import load_dotenv



warnings.filterwarnings("ignore", message="FP16 is not supported on CPU*")

# ElevenLabs config
ELEVENLABS_API_KEY = "sk_30dfc0862228d806a2ec6e6b3507d82342edbd6498850787"
ELEVENLABS_VOICE_ID = "KUyRrVDjGxd32dlMuV24"

# Gemini config
# genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# api_key = os.getenv("GEMINI_API_KEY")
# if not api_key:
#     raise EnvironmentError("❌ 'GEMINI_API_KEY' is not set in environment variables.")
# genai.configure(api_key=api_key)

load_dotenv()  # Load the .env file

genai.configure(api_key=os.environ["GEMINI_API_KEY"])


generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)



with open('PromptNew2.txt', "r", encoding="utf-8") as file:
    prompt = file.read()

chat_session = model.start_chat(
    history=[
        {
            "role": "user",
            "parts": [
                prompt
            ],
        },
    ]
)

survey_data = {}

def speak(text, lang="tr"):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.75,
            "similarity_boost": 0.75
        }
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        temp_file = os.path.join(tempfile.gettempdir(), f"response_{int(time.time())}.mp3")
        with open(temp_file, "wb") as f:
            f.write(response.content)
        pygame.mixer.init()
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.quit()
        os.remove(temp_file)
    else:
        raise Exception(f"ElevenLabs API error: {response.text}")


# Whisper + VAD settings
CHUNK_DURATION_MS = 30
RATE = 16000
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RECORD_TIMEOUT = 1
VAD_MODE = 1

vad = webrtcvad.Vad(VAD_MODE)
pa = pyaudio.PyAudio()
stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)

whisper_model = whisper.load_model("small") # base small medium large 

print("🎤 AI Survey Started (Say 'exit' or 'kapat' to stop)")

frames = []
ring_buffer = collections.deque(maxlen=int(RATE / CHUNK_SIZE * RECORD_TIMEOUT))
recording = False
last_question = None

try:
    while True:
        frame = stream.read(CHUNK_SIZE)
        if vad.is_speech(frame, RATE):
            if not recording:
                print("Speak!")
                recording = True
                frames = []
            ring_buffer.clear()
            frames.append(frame)
        elif recording:
            ring_buffer.append(frame)
            if len(ring_buffer) >= ring_buffer.maxlen:
                print("Silence detected!!!")
                frames.extend(ring_buffer)
                recording = False

                filename = "temp.wav"
                with wave.open(filename, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(pa.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(frames))

                # Transcribe
                with torch.no_grad():
                    result = whisper_model.transcribe(filename, language="tr")
                user_input = result['text'].strip()

                # Skip if transcription is empty
                if not user_input:
                    print("⚠️ No speech detected. Skipping...")
                    continue

                print(f"📝 You: {user_input}")

                if user_input.lower() in ["exit", "kapat", "bitir"]:
                    speak("Anket sonlandırıldı. Katılımınız için teşekkür ederiz.")
                    break

                response = chat_session.send_message(user_input)
                bot_response = response.text
                print(f"🤖 Bot: {bot_response}")
                speak(bot_response)

                

                last_question = bot_response

                if "Teşekkür ederiz! Anketimiz sona erdi" in bot_response or "katılımınız için teşekkür ederiz" in bot_response:
                    break

except KeyboardInterrupt:
    print(" Survey manually stopped.")
    speak("Anket kullanıcı tarafından sonlandırıldı.")

finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()
