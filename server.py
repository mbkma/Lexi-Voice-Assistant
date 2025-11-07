#!/usr/bin/env python3
"""
Lexi Voice Assistant - Server Component
Handles all heavy processing: speech recognition, LLM inference, and TTS generation
"""

import os
import sys
import json
import socket
import threading
import warnings
import subprocess
import struct
import wave
import io

# Suppress ALSA/JACK warnings
os.environ['ALSA_CARD'] = 'default'
if sys.platform == 'linux':
    stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')

from vosk import Model, KaldiRecognizer

if sys.platform == 'linux':
    sys.stderr.close()
    sys.stderr = stderr

from llama_cpp import Llama

warnings.filterwarnings('ignore', category=RuntimeWarning, module='llama_cpp')

from main import (
    load_config, get_model_paths, download_vosk_model,
    download_llama_model, download_piper_model, detect_gpu,
    SAMPLE_RATE, MESSAGES
)

# Server configuration
DEFAULT_HOST = '0.0.0.0'
DEFAULT_PORT = 5555
CHUNK_SIZE = 4096


class VoiceAssistantServer:
    """Server that handles all heavy processing for voice assistant"""

    def __init__(self, config, host=DEFAULT_HOST, port=DEFAULT_PORT):
        print("Initializing Voice Assistant Server...")

        self.config = config
        self.host = host
        self.port = port
        self.paths = get_model_paths(config)

        # Download models if needed
        print("Checking models...")
        download_vosk_model(
            self.paths["vosk_model_path"],
            self.paths["vosk_model_url"],
            self.paths["vosk_model_name"]
        )
        download_llama_model(
            self.paths["llama_model_path"],
            self.paths["llama_model_url"]
        )
        download_piper_model(
            self.paths["piper_model_path"],
            self.paths["piper_config_path"],
            self.paths["piper_model_url"],
            self.paths["piper_config_url"]
        )

        # Initialize models
        print("Loading speech recognition model...")
        self.vosk_model = Model(self.paths["vosk_model_path"])

        print("Loading language model...")
        has_gpu = detect_gpu()
        n_gpu_layers = 35 if has_gpu else 0
        n_threads = 4 if not has_gpu else 1
        print(f"LLM config: GPU layers={n_gpu_layers}, CPU threads={n_threads}")

        self.llm = Llama(
            model_path=self.paths["llama_model_path"],
            n_ctx=2048,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            use_mmap=True,
            use_mlock=False
        )

        # Server socket
        self.server_socket = None
        self.running = False

        print("Server initialization complete!\n")

    def recognize_speech(self, audio_data):
        """Recognize speech from audio data"""
        recognizer = KaldiRecognizer(self.vosk_model, SAMPLE_RATE)

        # Process audio chunks
        text = ""
        offset = 0

        while offset < len(audio_data):
            chunk = audio_data[offset:offset + CHUNK_SIZE]
            offset += CHUNK_SIZE

            if recognizer.AcceptWaveform(chunk):
                result = json.loads(recognizer.Result())
                text = result.get('text', '')
                if text:
                    break

        # Get final result if no text yet
        if not text:
            result = json.loads(recognizer.FinalResult())
            text = result.get('text', '')

        return text.strip()

    def generate_llm_response(self, conversation_history, user_input):
        """Generate LLM response"""
        # Add user input to history
        conversation_history.append({"role": "user", "content": user_input})

        # Build prompt
        prompt = ""
        for msg in conversation_history[-6:]:  # Keep last 3 exchanges
            if msg["role"] == "user":
                prompt += f"[INST] {msg['content']} [/INST] "
            else:
                prompt += msg["content"] + " "

        # Generate response
        output = self.llm(
            prompt.strip(),
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            stop=["</s>", "[INST]", "[/INST]"],
            echo=False
        )

        response = output['choices'][0]['text'].strip()
        conversation_history.append({"role": "assistant", "content": response})

        return response, conversation_history

    def generate_speech(self, text):
        """Generate speech audio from text using Piper TTS"""
        try:
            # Use Piper to generate audio
            result = subprocess.run(
                ["piper", "--model", self.paths["piper_model_path"], "--output-raw"],
                input=text.encode('utf-8'),
                capture_output=True,
                check=True
            )

            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Error generating speech: {e}")
            return None
        except FileNotFoundError:
            print("Piper TTS not found. Install with: pip install piper-tts")
            return None

    def process_command(self, text, conversation_history):
        """Process special commands"""
        text_lower = text.lower()

        # English commands
        exit_words_en = ['exit', 'quit', 'goodbye', 'bye']
        clear_words_en = ['clear', 'reset', 'new conversation']

        # German commands
        exit_words_de = ['beenden', 'tschüss', 'auf wiedersehen']
        clear_words_de = ['löschen', 'zurücksetzen', 'neue konversation']

        if any(word in text_lower for word in exit_words_en + exit_words_de):
            return "exit", conversation_history
        elif any(word in text_lower for word in clear_words_en + clear_words_de):
            return "cleared", []

        return None, conversation_history

    def get_message(self, key):
        """Get message in configured language"""
        lang = self.config["language"]
        return MESSAGES[lang][key]

    def handle_client(self, client_socket, address):
        """Handle individual client connection"""
        print(f"Client connected from {address}")
        conversation_history = []

        try:
            while True:
                # Receive message type and data length
                header = client_socket.recv(8)
                if not header or len(header) < 8:
                    break

                msg_type = struct.unpack('!I', header[:4])[0]
                data_length = struct.unpack('!I', header[4:8])[0]

                # Receive data
                data = b''
                while len(data) < data_length:
                    chunk = client_socket.recv(min(4096, data_length - len(data)))
                    if not chunk:
                        break
                    data += chunk

                if not data:
                    break

                # Process based on message type
                # Type 1: Audio for recognition
                if msg_type == 1:
                    print(f"Received audio data ({len(data)} bytes) from {address}")
                    text = self.recognize_speech(data)
                    print(f"Recognized text: {text}")

                    # Send recognized text back
                    text_bytes = text.encode('utf-8')
                    response_header = struct.pack('!II', 2, len(text_bytes))
                    client_socket.sendall(response_header + text_bytes)

                # Type 3: Text for LLM processing
                elif msg_type == 3:
                    text = data.decode('utf-8')
                    print(f"Processing text: {text}")

                    # Check for commands
                    command, conversation_history = self.process_command(text, conversation_history)

                    if command == "exit":
                        response_text = self.get_message("goodbye")
                        response_bytes = response_text.encode('utf-8')
                        response_header = struct.pack('!II', 4, len(response_bytes))
                        client_socket.sendall(response_header + response_bytes)

                        # Generate and send speech
                        audio_data = self.generate_speech(response_text)
                        if audio_data:
                            audio_header = struct.pack('!II', 5, len(audio_data))
                            client_socket.sendall(audio_header + audio_data)
                        break

                    elif command == "cleared":
                        response_text = self.get_message("cleared")
                    else:
                        # Generate LLM response
                        print("Generating LLM response...")
                        response_text, conversation_history = self.generate_llm_response(
                            conversation_history, text
                        )
                        print(f"Generated response: {response_text}")

                    # Send text response
                    response_bytes = response_text.encode('utf-8')
                    response_header = struct.pack('!II', 4, len(response_bytes))
                    client_socket.sendall(response_header + response_bytes)

                    # Generate and send speech audio
                    print("Generating speech...")
                    audio_data = self.generate_speech(response_text)
                    if audio_data:
                        audio_header = struct.pack('!II', 5, len(audio_data))
                        client_socket.sendall(audio_header + audio_data)
                        print(f"Sent audio data ({len(audio_data)} bytes)")

                # Type 6: Get greeting message
                elif msg_type == 6:
                    greeting = self.get_message("greeting")
                    greeting_bytes = greeting.encode('utf-8')
                    response_header = struct.pack('!II', 4, len(greeting_bytes))
                    client_socket.sendall(response_header + greeting_bytes)

                    # Generate and send greeting audio
                    audio_data = self.generate_speech(greeting)
                    if audio_data:
                        audio_header = struct.pack('!II', 5, len(audio_data))
                        client_socket.sendall(audio_header + audio_data)

        except Exception as e:
            print(f"Error handling client {address}: {e}")
            import traceback
            traceback.print_exc()

        finally:
            print(f"Client {address} disconnected")
            client_socket.close()

    def start(self):
        """Start the server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)

        self.running = True
        print(f"Server listening on {self.host}:{self.port}")
        print("Waiting for clients...\n")

        try:
            while self.running:
                client_socket, address = self.server_socket.accept()

                # Handle each client in a separate thread
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, address),
                    daemon=True
                )
                client_thread.start()

        except KeyboardInterrupt:
            print("\nShutting down server...")

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        print("Server shut down successfully.")


def main():
    """Entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Lexi Voice Assistant Server")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port (default: 5555)")
    args = parser.parse_args()

    print("=" * 60)
    print("LEXI VOICE ASSISTANT SERVER")
    print("=" * 60)
    print()

    # Load configuration
    config = load_config()

    if not config.get("initialized", False):
        print("Error: Assistant not initialized. Please run main.py --init first.")
        sys.exit(1)

    print(f"Configuration: Language={config['language'].upper()}, Model={config['llama_model'].upper()}")
    print()

    # Start server
    try:
        server = VoiceAssistantServer(config, args.host, args.port)
        server.start()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
