import os
import tempfile
import time
import json
import requests
import threading
import queue
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
# --- Voice & Language ---
# The index of your virtual speaker. Find with: pactl list short sinks
SPEAKER_DEVICE_INDEX = 1102 # Use the index you found
# --- Voice & Language ---
# The language for STT and TTS, e.g., "tr-TR", "en-US"
LANGUAGE_CODE = "tr-TR"
# Sample rate for all audio processing. Google prefers 16000. Asterisk will handle resampling.
SAMPLE_RATE = 16000
# Path where Asterisk will save live call recordings.
# Ensure this directory exists and has correct permissions for Asterisk and this script.
LIVE_RECORDING_PATH = "/var/spool/asterisk/recording"

# --- API & Services ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Set Google Cloud credentials
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google-tts-key.json"

# --- Asterisk ARI Configs ---
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
# Load your prompt file
with open('PromptNew2.txt', "r", encoding="utf-8") as file:
    prompt = file.read()


# === NEW: Asterisk Live Audio Streamer ===
# === NEW: Asterisk Live Audio Streamer (with retry logic) ===
class AsteriskLiveAudioStreamer(threading.Thread):
    """
    Waits for a live recording file to appear, then tails it and pushes
    the audio chunks into multiple consumer queues.
    """
    def __init__(self, recording_path, consumer_queues):
        super().__init__()
        self.recording_path = recording_path
        self.consumer_queues = consumer_queues
        self._stop_event = threading.Event()
        self.chunk_size = int(SAMPLE_RATE * 30 / 1000) * 2  # 30ms chunks, 16-bit audio

    def run(self):
        # --- NEW: Wait actively for the file to be created ---
        timeout_seconds = 5
        start_time = time.time()
        file_exists = os.path.exists(self.recording_path)
        
        while not file_exists and not self._stop_event.is_set():
            if time.time() - start_time > timeout_seconds:
                print(f"❌ FATAL: Timed out after {timeout_seconds}s waiting for recording file: {self.recording_path}")
                return # Exit the thread if file never appears
            time.sleep(0.1) # Wait 100ms before checking again
            file_exists = os.path.exists(self.recording_path)
        
        if self._stop_event.is_set():
            return # Exit if stop was called while waiting
        
        print(f"✅ File found! Starting to stream audio from {self.recording_path}")
        try:
            with open(self.recording_path, "rb") as f:
                while not self._stop_event.is_set():
                    chunk = f.read(self.chunk_size)
                    if chunk:
                        for q in self.consumer_queues:
                            q.put(chunk)
                    else:
                        # If no data, wait a bit for more to be written
                        time.sleep(0.01)
        except Exception as e:
            print(f"❌ Error during audio streaming from file: {e}")
        finally:
            print(f"⏹️ Audio streamer for {self.recording_path} stopped.")

    def stop(self):
        self._stop_event.set()

# === NEW: Speaker Player ===
# === NEW Speaker Player (with hardware matching attempt) ===
# === NEW Speaker Player (with hardware matching attempt) ===
class SpeakerPlayer(threading.Thread):
    """
    Plays audio from a queue to the local speakers.
    This version attempts to match the hardware's native sample rate.
    """
    def __init__(self, audio_queue, device_index=None):
        super().__init__()
        self.audio_queue = audio_queue
        self._stop_event = threading.Event()
        self.p = pyaudio.PyAudio()

        # --- Let's try to match the hardware's expected sample rate ---
        HARDWARE_RATE = 48000
        # The audio from Asterisk is 16000Hz, so we're playing it on a 48000Hz stream.
        # This will make it sound sped up, BUT our goal here is to see if it fixes the ERROR.
        
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,  # Asterisk sends 16-bit audio
                channels=1,
                rate=HARDWARE_RATE, # Using 48000Hz
                output=True,
                output_device_index=device_index
            )
            print(f"🔊 Speaker player initialized on device {device_index} at {HARDWARE_RATE}Hz.")
        except Exception as e:
            print(f"❌ FAILED to open speaker device {device_index} at {HARDWARE_RATE}Hz: {e}")
            self.stream = None # Ensure stream is None on failure
            self.p.terminate()

    def run(self):
        if not self.stream:
            print("🔊 Speaker player not initialized, thread exiting.")
            return

        while not self._stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=1)
                self.stream.write(chunk)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Speaker player error: {e}")
                break
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        print("Speaker player stopped.")

    def stop(self):
        self._stop_event.set()

# === Modified Google TTS ===
def speak_and_prepare_for_asterisk(text, lang=LANGUAGE_CODE):
    """
    Generates speech using Google TTS and converts it to a format
    Asterisk can play directly (16-bit PCM, 8000Hz WAV).
    """
    try:
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=lang,
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        # Generate raw linear16 audio data
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE
        )
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        # Asterisk usually prefers 8000 Hz for .wav files in /var/lib/asterisk/sounds
        # We will convert it using a temporary file.
        temp_raw_path = os.path.join(tempfile.gettempdir(), "tts_temp.raw")
        final_wav_path = "/var/lib/asterisk/sounds/ai_response.wav"

        with open(temp_raw_path, "wb") as out:
            out.write(response.audio_content)

        # Use ffmpeg to resample and create the final WAV header
        # -f s16le: input format is 16-bit signed little-endian (LINEAR16)
        # -ar 8000: output audio rate 8kHz
        # -ac 1: output audio channels 1
        ffmpeg_command = (
            f"ffmpeg -y -f s16le -ar {SAMPLE_RATE} -ac 1 -i {temp_raw_path} "
            f"-ar 8000 -ac 1 {final_wav_path}"
        )
        os.system(ffmpeg_command)

        # The sound name for ARI is the filename without extension
        return "ai_response"
    except Exception as e:
        print(f"❌ Google TTS Error: {e}")
        return None
    
# === MODIFIED: VAD-Based Playback Monitor (from Queue) ===
class InterruptPlaybackMonitor(threading.Thread):
    """
    Listens to an audio queue for speech activity to detect interruptions.
    """
    def __init__(self, channel_id, audio_queue, aggressiveness=2, duration=0.5):
        super().__init__()
        self.channel_id = channel_id
        self.audio_queue = audio_queue
        self.vad = webrtcvad.Vad(aggressiveness)
        self.rate = SAMPLE_RATE
        self.chunk_duration_ms = 30  # VAD supports 10, 20, or 30 ms
        self.chunk_size = int(self.rate * self.chunk_duration_ms / 1000) * 2
        self.duration_to_interrupt_s = duration
        self._stop_event = threading.Event()
        self.interrupted = threading.Event()

    def run(self):
        print(" Muting for interruption...")
        speech_chunks_needed = int(self.duration_to_interrupt_s * 1000 / self.chunk_duration_ms)
        continuous_speech_chunks = 0
        try:
            while not self._stop_event.is_set():
                try:
                    frame = self.audio_queue.get(timeout=0.1)
                    is_speech = self.vad.is_speech(frame, self.rate)
                    if is_speech:
                        continuous_speech_chunks += 1
                        if continuous_speech_chunks >= speech_chunks_needed:
                            print(" Voice detected — stopping playback!")
                            self.stop_playback()
                            self.interrupted.set()
                            break
                    else:
                        continuous_speech_chunks = 0
                except queue.Empty:
                    continue
        finally:
            print("Interruption monitor stopped.")

    def stop_playback(self):
        print("🛑 Sending stop playback command to ARI...")
        try:
            requests.delete(f"{BASE_URL}/playbacks/sound:ai_response", auth=HTTPBasicAuth(ARI_USER, ARI_PASSWORD))
            # Fallback for older Asterisk versions
            requests.delete(f"{BASE_URL}/channels/{self.channel_id}/play", auth=HTTPBasicAuth(ARI_USER, ARI_PASSWORD))
        except requests.RequestException as e:
            print(f"Failed to stop playback via ARI: {e}")

    def stop(self):
        self._stop_event.set()

# === NEW: Google STT Streamer (from Queue) ===
class GoogleStreamer:
    """
    Manages Google's Streaming Speech-to-Text from an audio queue.
    """
    def __init__(self, audio_queue, language_code=LANGUAGE_CODE):
        self.audio_queue = audio_queue
        self.language_code = language_code
        self.client = speech.SpeechClient()
        self.config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            language_code=self.language_code,
            max_alternatives=1,
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
                if chunk is None:
                    return
                yield speech.StreamingRecognizeRequest(audio_content=chunk)
            except queue.Empty:
                continue

    def listen(self, silence_timeout=3):
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
            if not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript.strip()
            
            if result.is_final:
                print(f"🗣️  Final Transcript: {transcript}")
                return transcript
            else:
                last_speech_time = time.time()
                print(f" interim: {transcript}")
        return None

    def close(self):
        self._closed.set()


# === REWRITTEN: Main Conversation Logic ===
def interact_with_user(channel_id, recording_file):
    """Handles the main conversation flow for a single call."""
    print(f"🚀 Interaction starting for channel {channel_id}. Recording at: {recording_file}")

    # 1. Create queues for distributing audio
    stt_queue = queue.Queue()
    vad_queue = queue.Queue()
    # speaker_queue = queue.Queue() # You can leave this in, but comment out the player below if you have audio issues

    # 2. Start the components that consume audio
    # The streamer is now started AFTER we know the recording has begun
    audio_streamer = AsteriskLiveAudioStreamer(recording_file, [stt_queue, vad_queue])
# In interact_with_user()
    # speaker_player = SpeakerPlayer(speaker_queue, device_index=SPEAKER_DEVICE_INDEX)
    stt_client = GoogleStreamer(stt_queue)

    audio_streamer.start()
    # speaker_player.start() # <<< AND THIS ONE

    chat_session = model.start_chat(history=[{"role": "user", "parts": [prompt]}])

    try:
        # The main loop is the same, but the recording is already active.
        while True:
            # 3. Listen for user input
            user_text = stt_client.listen()
            # ... (the rest of the loop remains exactly the same as before) ...
            if not user_text:
                print("⚠️ No input detected or silence, listening again.")
                continue

            # 4. Process with Gemini
            print(f"📝 User said: {user_text}")
            response = chat_session.send_message(user_text)
            reply = response.text.strip()
            print(f"🤖 Gemini: {reply}")

            # 5. Speak the response and allow for interruption
            media_id = speak_and_prepare_for_asterisk(reply)
            if media_id:
                interrupt_monitor = InterruptPlaybackMonitor(channel_id, vad_queue)
                interrupt_monitor.start()

                playback_response = requests.post(
                    f"{BASE_URL}/channels/{channel_id}/play",
                    params={"media": f"sound:{media_id}"},
                    auth=HTTPBasicAuth(ARI_USER, ARI_PASSWORD)
                )

                while not interrupt_monitor.interrupted.is_set():
                    time.sleep(0.2)
                    try:
                        res = requests.get(f"{BASE_URL}/playbacks/sound:{media_id}", auth=HTTPBasicAuth(ARI_USER, ARI_PASSWORD))
                        if res.status_code != 200:
                            break
                    except requests.RequestException:
                        break

                interrupt_monitor.stop()
                interrupt_monitor.join()

            stop_words = ["exit", "kapat", "bitir"]
            closing_phrases = ["teşekkür ederiz", "anketimiz sona erdi"]
            if any(word in user_text.lower() for word in stop_words) or \
               any(phrase in reply.lower() for phrase in closing_phrases):
                print("📴 Ending call based on conversation.")
                break

    except Exception as e:
        print(f"❌ An error occurred in the interaction loop: {e}")
    finally:
        # 6. Cleanup
        print(f"🧹 Cleaning up resources for channel {channel_id}")
        stt_client.close()
        audio_streamer.stop()
        # speaker_player.stop() # <<< AND THIS ONE
        audio_streamer.join()
        # speaker_player.join() # <<< AND THIS ONE

        requests.delete(f"{BASE_URL}/channels/{channel_id}", auth=HTTPBasicAuth(ARI_USER, ARI_PASSWORD))

        if os.path.exists(recording_file):
            os.remove(recording_file)
            print(f"Deleted recording file: {recording_file}")

            
# === WebSocket Event Handlers (Modified for new flow) ===
# === WebSocket Event Handlers (Modified for new flow) ===
# === WebSocket Event Handlers (Final Version) ===
def on_message(ws, message):
    event = json.loads(message)
    event_type = event.get('type')
    print(f"📡 Event: {event_type}")

    if event_type == 'StasisStart':
        channel_id = event['channel']['id']
        print(f"Channel {channel_id} entered Stasis. Answering and starting recording.")

        # --- NEW: Use a simple name for the recording ---
        # We will let Asterisk decide the exact final filename and path.
        simple_name = f"live_rec_{channel_id}"

        # Answer the call
        requests.post(f"{BASE_URL}/channels/{channel_id}/answer", auth=HTTPBasicAuth(ARI_USER, ARI_PASSWORD))

        # Start recording using the simple name
        requests.post(
            f"{BASE_URL}/channels/{channel_id}/record",
            params={
                "name": simple_name,  # <-- Using the simple name
                "format": "sln16",
                "ifExists": "overwrite"
            },
            auth=HTTPBasicAuth(ARI_USER, ARI_PASSWORD)
        )

    elif event_type == 'RecordingStarted':
        recording = event.get('recording', {})
        channel_id = recording.get('target_uri').split(':')[1]
        
        # The actual filename will be the 'name' from the event, plus the format extension.
        # It should be saved in LIVE_RECORDING_PATH by default.
        final_filename = f"{recording.get('name')}.sln16"
        full_path = os.path.join(LIVE_RECORDING_PATH, final_filename)

        if channel_id and final_filename:
            print(f"✅ Recording has started for channel {channel_id}.")
            print(f"   Expected file at: {full_path}")
            
            # Launch the main logic with the full, expected path
            threading.Thread(
                target=interact_with_user,
                args=(channel_id, full_path)
            ).start()

    elif event_type == 'RecordingFailed':
        print(f"❌ Recording failed: {event.get('recording')}")

    elif event_type == 'StasisEnd':
        print("📞 Call ended and channel destroyed.")

def on_open(ws):
    print("✅ WebSocket connected to ARI")
    originate_call()

def on_error(ws, error):
    print(f"❗ WebSocket Error: {error}")

def on_close(ws, *args):
    print("🔌 WebSocket closed")


# === Outbound Call Originate ===
def originate_call():
    """Initiates an outbound call to a specified endpoint."""
    print("📞 Attempting to originate an outbound call...")
    url = f'{BASE_URL}/channels'
    # MAKE SURE this is the correct endpoint you want to call
    data = {
        "endpoint": "PJSIP/7001", 
        "extension": "s",  # Send the call directly into the Stasis app
        "context": "default", # Use a simple context
        "priority": "1",
        "app": ARI_APP,
        "callerId": "AI Bot"
    }
    try:
        response = requests.post(url, data=data, auth=HTTPBasicAuth(ARI_USER, ARI_PASSWORD))
        response.raise_for_status() # This will raise an error for bad status codes (4xx or 5xx)
        print("✅ Outbound call successfully initiated.")
    except requests.RequestException as e:
        print(f"❌ Call failed: {e}")
        if e.response is not None:
            print(f"Response body: {e.response.text}")

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
    # The original script had an originate_call() on open. 
    # This version is designed to RECEIVE calls into the ARI app.
    # If you still want to originate the call, you can add the originate_call() function
    # back and call it in on_open().
    print("Starting AI Agent. Waiting for incoming calls...")
    start_websocket()    
