import requests
import websocket
import threading
import json
import time
from requests.auth import HTTPBasicAuth

# ARI configuration
ARI_USER = 'ai'
ARI_PASSWORD = 'ai_secret'
ARI_HOST = 'localhost'
ARI_PORT = 8088
ARI_APP = 'aiagent'

# REST API base URL
BASE_URL = f'http://{ARI_HOST}:{ARI_PORT}/ari'

# Function to handle incoming WebSocket messages
def on_message(ws, message):
    event = json.loads(message)
    print(f"Received event: {event['type']}")
    # Add your event handling logic here

# Function to handle WebSocket errors
def on_error(ws, error):
    print(f"WebSocket error: {error}")

# Function to handle WebSocket closure
def on_close(ws, close_status_code, close_msg):
    print("WebSocket connection closed")

# Function to handle WebSocket opening
def on_open(ws):
    print("WebSocket connection established")

# Function to start the WebSocket connection
def start_websocket():
    ws_url = f'ws://{ARI_HOST}:{ARI_PORT}/ari/events?app={ARI_APP}&api_key={ARI_USER}:{ARI_PASSWORD}'
    ws = websocket.WebSocketApp(ws_url,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever()

# Function to list active channels (example REST API call)
def list_channels():
    url = f'{BASE_URL}/channels'
    response = requests.get(url, auth=HTTPBasicAuth(ARI_USER, ARI_PASSWORD))
    if response.status_code == 200:
        channels = response.json()
        print("Active channels:")
        for channel in channels:
            print(f" - {channel['name']}")
    else:
        print(f"Failed to retrieve channels: {response.status_code}")

if __name__ == "__main__":
    # Start WebSocket in a separate thread
    ws_thread = threading.Thread(target=start_websocket)
    ws_thread.daemon = True
    ws_thread.start()

    # Periodically list active channels
    try:
        while True:
            list_channels()
            time.sleep(10)
    except KeyboardInterrupt:
        print("Exiting...")
