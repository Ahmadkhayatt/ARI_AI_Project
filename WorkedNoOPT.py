#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import wave
import pwd
import grp

import os
import tempfile
import time
import json
import requests
import threading
import queue
import wave  # Use the built-in wave library
from dotenv import load_dotenv
from google.cloud import texttospeech, speech
import google.generativeai as genai
from requests.auth import HTTPBasicAuth
import websocket
import webrtcvad

# --- IMPORTANT ---
# This script is designed to be run with sudo to have the necessary permissions
# to write audio files directly into the Asterisk sounds directory.
# Example: sudo /path/to/your/venv/bin/python /path/to/this/script.py

# === Load environment variables ===
# Make sure you have a .env file with your GEMINI_API_KEY
load_dotenv()

# Set Google Cloud credentials environment variable
# Make sure the 'google-tts-key.json' file is in the same directory as this script.
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google-tts-key.json"

# === Configs ===
# --- Voice & Language ---
LANGUAGE_CODE = "tr-TR"
SAMPLE_RATE = 16000  # Rate for STT and live audio processing

# --- API & Services ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Asterisk ARI Configs ---
ARI_USER = 'ai'
ARI_PASSWORD = 'ai_secret'
ARI_HOST = 'localhost'
ARI_PORT = 8088
ARI_APP = 'aiagent'
BASE_URL = f'http://{ARI_HOST}:{ARI_PORT}/ari'

# --- Paths ---
# Directory for Asterisk to save live recordings.
# Ensure this has the correct permissions (sudo chown -R asterisk:asterisk /var/spool/asterisk/recording)
LIVE_RECORDING_PATH = "/var/spool/asterisk/recording"
# The single, overwritable TTS file in the main Asterisk sounds directory.
TTS_SOUND_FILE_PATH = "/var/lib/asterisk/sounds/ai_agent_response.wav"
# The simple name Asterisk will use to play the sound.
TTS_SOUND_ID = "ai_agent_response"


# === Gemini Setup ===
try:
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
    # Load your prompt file
    with open('PromptNew2.txt', "r", encoding="utf-8") as file:
        prompt = file.read()
except Exception as e:
    print(f"FATAL: Could not initialize GenerativeAI. Check API Key and prompt file. Error: {e}")
    exit()


# === THE DEFINITIVE TTS FUNCTION ===
# The new, corrected TTS function.
def speak_and_prepare_for_asterisk(text, lang="tr-TR"):
    """
    Generates speech via Google TTS, writes it out as a real WAV file
    (16-bit LE PCM @ 8000 Hz), and makes sure Asterisk can read it.
    Returns the base media ID for ARI playback.
    """
    try:
        # 1) Call Google TTS
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=lang,
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=8000
        )
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        pcm = response.audio_content  # raw 16-bit LE PCM @ 8 kHz

        # 2) Wrap it in a WAV file
        wav_path = "/var/lib/asterisk/sounds/en/ai_agent_response.wav"
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)         # 16-bit = 2 bytes
            wf.setframerate(8000)      # must match the TTS sample_rate_hertz
            wf.writeframes(pcm)

        # 3) chown/chmod so the asterisk user can read it
        a_uid = pwd.getpwnam("asterisk").pw_uid
        a_gid = grp.getgrnam("asterisk").gr_gid
        os.chown(wav_path, a_uid, a_gid)
        os.chmod(wav_path, 0o644)

        print(f"  🔊 WAV saved and permissions set: {wav_path}")
        return "ai_agent_response"

    except Exception as e:
        print(f"❌ TTS+WAV error: {e}")
        return None
    
# === Asterisk Live Audio Streamer ===
class AsteriskLiveAudioStreamer(threading.Thread):

    

    """
    Waits for a live recording file to appear, then tails it and pushes
    the audio chunks into consumer queues.
    """
    def __init__(self, recording_path, consumer_queues):
        super().__init__()
        self.recording_path = recording_path
        self.consumer_queues = consumer_queues
        self._stop_event = threading.Event()
        self.chunk_size = int(SAMPLE_RATE * 30 / 1000) * 2

    def run(self):
        timeout_seconds = 5
        start_time = time.time()
        file_exists = False
        while not file_exists and not self._stop_event.is_set():
            if time.time() - start_time > timeout_seconds:
                print(f"❌ FATAL: Timed out waiting for recording file: {self.recording_path}")
                return
            file_exists = os.path.exists(self.recording_path)
            if not file_exists:
                time.sleep(0.1)

        print(f"✅ File found! Starting to stream audio from {self.recording_path}")
        try:
            with open(self.recording_path, "rb") as f:
                while not self._stop_event.is_set():
                    chunk = f.read(self.chunk_size)
                    if chunk:
                        for q in self.consumer_queues:
                            q.put(chunk)
                    else:
                        time.sleep(0.01)
        except Exception as e:
            print(f"❌ Error during audio streaming from file: {e}")
        finally:
            print(f"⏹️ Audio streamer for {self.recording_path} stopped.")

    def stop(self):
        self._stop_event.set()


# === Google STT Streamer (from Queue) ===
class GoogleStreamer:
    """ Manages Google's Streaming Speech-to-Text from an audio queue. """
    def __init__(self, audio_queue, language_code=LANGUAGE_CODE):
        self.audio_queue = audio_queue
        self.language_code = language_code
        self.client = speech.SpeechClient()
        self.config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            language_code=self.language_code,
            enable_automatic_punctuation=True,
        )
        self.streaming_config = speech.StreamingRecognitionConfig(
            config=self.config, interim_results=False
        )
        self._closed = threading.Event()

    def _generator(self):
        while not self._closed.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.5)
                yield speech.StreamingRecognizeRequest(audio_content=chunk)
            except queue.Empty:
                continue

    def listen(self, silence_timeout=7):
        print("🎤 Listening for user input via Google STT...")
        audio_generator = self._generator()
        responses = self.client.streaming_recognize(
            config=self.streaming_config, requests=audio_generator
        )
        last_speech_time = time.time()
        for response in responses:
            if not response.results:
                if time.time() - last_speech_time > silence_timeout:
                    print("...Silence timeout reached.")
                    break
                continue
            result = response.results[0]
            if result.is_final:
                transcript = result.alternatives[0].transcript.strip()
                print(f"🗣️  Final Transcript: {transcript}")
                return transcript
        return None

    def close(self):
        self._closed.set()


# === Main Conversation Logic ===
def interact_with_user(channel_id, recording_file, recording_name):
    global channel_id1 
    """ Handles the main conversation flow for a single call. """
    print(f"🚀 Interaction starting for channel {channel_id}.")
    stt_queue = queue.Queue()
    # The VAD queue is not used since interruption is disabled, but we leave the plumbing.
    vad_queue = queue.Queue()

    audio_streamer = AsteriskLiveAudioStreamer(recording_file, [stt_queue, vad_queue])
    stt_client = GoogleStreamer(stt_queue)

    audio_streamer.start()
    chat_session = model.start_chat(history=[{"role": "user", "parts": [prompt]}])
    stop_words = ["exit", "kapat", "bitir"]

    try:
        while not audio_streamer._stop_event.is_set():
            user_text = stt_client.listen()
            if not user_text:
                print("⚠️ No input detected, listening again.")
                continue

            print(f"📝 User said: {user_text}")
            if any(word in user_text.lower() for word in stop_words):
                print("📴 Ending call based on stop word.")
                break

            response = chat_session.send_message(user_text)
            reply = response.text.strip()
            print(f"🤖 Gemini: {reply}")

            # === Playback Section ===
            # This simplified version disables interruptions to ensure playback works first.
            # PASTE THIS NEW BLOCK INTO YOUR main.py

    # === NEW, W# PASTE THIS NEW DIAGNOSTIC BLOCK INTO YOUR main.py

            # === FINAL, CORRECT PLAYBACK LOGIC ===
            # The new, clean playback section for interact_with_user
            # === Final Working Playback ===

            reply_text = reply
            media_id = speak_and_prepare_for_asterisk(reply_text)

            if media_id:
                try:
                    print(f"  [INFO] Asking ARI to play 'sound:{media_id}'")
                    url = f"{BASE_URL}/channels/{channel_id}/play"
                    print("DEBUG → URL:", url)
                    print("DEBUG → params:", {"media":f"sound:{media_id}"})
                   
                    print("channel id : ", channel_id)


                    # response = requests.post(
                    #     url,
                    #     params={"media": f"sound:{media_id}.wav"},
                    #     auth=HTTPBasicAuth(ARI_USER, ARI_PASSWORD),
                    #     timeout=5
                    # )

                    stop_resp = requests.post(
                        f"{BASE_URL}/recordings/live/{recording_name}/stop",
                        auth=HTTPBasicAuth(ARI_USER, ARI_PASSWORD)
                    )
                    print("Stopped recording →", stop_resp.status_code, stop_resp.text)

                    time.sleep(1)
                    response = requests.post(
                        f"{BASE_URL}/channels/{channel_id}/play",
                        params={"media": f"sound:ai_agent_response" , "replaceMedia": True},
                        auth=HTTPBasicAuth(ARI_USER, ARI_PASSWORD)                        
                    )
                    response
                    print("DEBUG → Response status:", response.status_code)
                    print("DEBUG → Response body:", response.text)
                    print(f"  [INFO] ARI responded with status: {response.status_code}")
                    # Wait for the audio to play
                    time.sleep(4)
                    requests.post(
                    f"{BASE_URL}/channels/{channel_id}/record",
                    params={
                        "name": recording_name,
                        "format": "sln16",
                        "ifExists": "overwrite"
                    },
                    auth=HTTPBasicAuth(ARI_USER, ARI_PASSWORD)
                )

                except requests.RequestException as e:
                    print(f"  [ERROR] The request to Asterisk failed: {e}")
            print("--- Exiting Playback Section ---")
            closing_phrases = ["teşekkür ederim", "anketimiz sona erdi"]
            if any(phrase in reply.lower() for phrase in closing_phrases):
                print("📴 Ending call based on Gemini response.")
                break

    except Exception as e:
        print(f"❌ An error occurred in the interaction loop: {e}")
    finally:
        print(f"🧹 Cleaning up resources for channel {channel_id}")
        stt_client.close()
        audio_streamer.stop()
        audio_streamer.join()
        
        requests.delete(f"{BASE_URL}/channels/{channel_id}", auth=HTTPBasicAuth(ARI_USER, ARI_PASSWORD))
        if os.path.exists(recording_file):
            try:
                os.remove(recording_file)
                print(f"🗑️ Deleted recording file: {recording_file}")
            except OSError as e:
                print(f"   Error deleting recording file: {e}")


# === WebSocket Event Handlers ===
def on_message(ws, message):
    global channel_id1
    event = json.loads(message)
    event_type = event.get('type')
    print(f"📡 Event: {event_type}")

    if event_type == 'StasisStart':
        channel_id = event['channel']['id']
        print(f"Channel {channel_id} entered Stasis. Answering and starting recording.")
        requests.post(f"{BASE_URL}/channels/{channel_id}/answer",
                         auth=HTTPBasicAuth(ARI_USER, ARI_PASSWORD))
        
        simple_name = f"live_rec_{channel_id}"
        print("for debugging ",simple_name)
      
        
        requests.post(
            f"{BASE_URL}/channels/{channel_id}/record",
            params={"name": simple_name, "format": "sln16", "ifExists": "overwrite"},
            auth=HTTPBasicAuth(ARI_USER, ARI_PASSWORD)
        )
       
        
#         requests.post(
#     f"{BASE_URL}/channels/{channel_id}/mixMonitor",
#     params={
#         # write a WAV in the same directory, using your simple_name
#         "filename": f"{LIVE_RECORDING_PATH}/{simple_name}.wav",
#         # 'b' = both directions, background mode (non-blocking)
#         "options": "b"
#     },
#     auth=HTTPBasicAuth(ARI_USER, ARI_PASSWORD)
# )

    # In on_message function...
    elif event_type == 'RecordingStarted':
        recording = event.get('recording', {})
        channel_id = recording.get('target_uri').split(':')[1]
        recording_name = recording.get('name') # Get the recording name
        final_filename = f"{recording_name}.sln16"
        full_path = os.path.join(LIVE_RECORDING_PATH, final_filename)
        if channel_id and final_filename:
            print(f"   Expected file at: {full_path}")
            # This is the changed line. Add recording_name to the arguments.
            threading.Thread(target=interact_with_user, args=(channel_id, full_path, recording_name)).start()
        elif event_type == 'StasisEnd':
            print("📞 Call ended and channel destroyed.")


# === Outbound Call Originate ===
def originate_call():
    """Initiates an outbound call to a specified endpoint."""
    print("📞 Attempting to originate an outbound call...")
    # NOTE: The extension should just be 's' and the context should be your app's context
    # This is defined in Asterisk's dialplan (extensions.conf)
    data = {
        "endpoint": "PJSIP/7001",
        "extension": "s",
        "context": "ai-survey", # This context must route to Stasis(aiagent)
        "priority": "1",
        "app": ARI_APP,
        "callerId": "AI Bot"
    }
    try:
        response = requests.post(f'{BASE_URL}/channels', data=data, auth=HTTPBasicAuth(ARI_USER, ARI_PASSWORD))
        response.raise_for_status()
        print("✅ Outbound call successfully initiated.")
    except requests.RequestException as e:
        print(f"❌ Call failed: {e}")
        if e.response is not None:
            print(f"Response body: {e.response.text}")

def on_open(ws):
    print("✅ WebSocket connected to ARI")
    originate_call()

def on_error(ws, error):
    print(f"❗ WebSocket Error: {error}")

def on_close(ws, *args):
    print("🔌 WebSocket closed")


# === App Entry ===
if __name__ == "__main__":
    print("--- Starting Definitive AI Agent ---")
    ws_url = f'ws://{ARI_HOST}:{ARI_PORT}/ari/events?app={ARI_APP}&api_key={ARI_USER}:{ARI_PASSWORD}'
    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()
