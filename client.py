#!/usr/bin/env python3
"""
Lexi Voice Assistant - Client Component
Handles only audio input/output and communicates with server for processing
"""

import os
import sys
import socket
import struct
import pyaudio
import wave
import io
import argparse

# Suppress ALSA/JACK warnings
os.environ['ALSA_CARD'] = 'default'
if sys.platform == 'linux':
    stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')

import pyaudio

if sys.platform == 'linux':
    sys.stderr.close()
    sys.stderr = stderr

# Audio configuration
SAMPLE_RATE = 16000
CHUNK_SIZE = 4096

# Server configuration
DEFAULT_SERVER_HOST = 'localhost'
DEFAULT_SERVER_PORT = 5555


class VoiceAssistantClient:
    """Client that handles audio I/O and communicates with server"""

    def __init__(self, server_host=DEFAULT_SERVER_HOST, server_port=DEFAULT_SERVER_PORT):
        print("Initializing Voice Assistant Client...")

        self.server_host = server_host
        self.server_port = server_port
        self.socket = None

        # Initialize audio
        self.audio = pyaudio.PyAudio()

        print("Client initialization complete!\n")

    def connect(self):
        """Connect to the server"""
        print(f"Connecting to server at {self.server_host}:{self.server_port}...")

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.server_host, self.server_port))
            print("Connected to server!\n")
            return True
        except Exception as e:
            print(f"Error connecting to server: {e}")
            return False

    def send_message(self, msg_type, data):
        """Send message to server"""
        header = struct.pack('!II', msg_type, len(data))
        self.socket.sendall(header + data)

    def receive_message(self):
        """Receive message from server"""
        # Receive header
        header = self.socket.recv(8)
        if not header or len(header) < 8:
            return None, None

        msg_type = struct.unpack('!I', header[:4])[0]
        data_length = struct.unpack('!I', header[4:8])[0]

        # Receive data
        data = b''
        while len(data) < data_length:
            chunk = self.socket.recv(min(4096, data_length - len(data)))
            if not chunk:
                break
            data += chunk

        return msg_type, data

    def record_audio(self, duration=5):
        """Record audio from microphone"""
        print("Listening... (speak now)")

        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )

        stream.start_stream()

        frames = []
        silence_chunks = 0
        max_silence = 30  # Stop after ~2 seconds of silence
        has_speech = False

        try:
            while True:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                frames.append(data)

                # Simple voice activity detection
                # Calculate RMS of audio chunk
                import array
                samples = array.array('h', data)
                rms = sum(s * s for s in samples) / len(samples)
                rms = rms ** 0.5

                # If RMS is above threshold, we have speech
                if rms > 300:  # Adjust threshold as needed
                    silence_chunks = 0
                    has_speech = True
                    print(".", end='', flush=True)
                else:
                    silence_chunks += 1
                    if has_speech and silence_chunks > max_silence:
                        break

                # Maximum recording time
                if len(frames) > (SAMPLE_RATE / CHUNK_SIZE) * 10:  # 10 seconds max
                    break

        except KeyboardInterrupt:
            pass
        finally:
            stream.stop_stream()
            stream.close()
            print()  # New line

        # Convert frames to bytes
        audio_data = b''.join(frames)
        return audio_data

    def play_audio(self, raw_audio_data):
        """Play raw PCM audio data from server"""
        try:
            # Raw audio from Piper is 16-bit PCM at 22050 Hz mono
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=22050,  # Piper output rate
                output=True
            )

            # Play audio in chunks
            chunk_size = 4096
            offset = 0
            while offset < len(raw_audio_data):
                chunk = raw_audio_data[offset:offset + chunk_size]
                stream.write(chunk)
                offset += chunk_size

            stream.stop_stream()
            stream.close()

        except Exception as e:
            print(f"Error playing audio: {e}")

    def get_greeting(self):
        """Get greeting from server"""
        # Send greeting request (type 6)
        self.send_message(6, b'')

        # Receive text response
        msg_type, data = self.receive_message()
        if msg_type == 4:
            greeting_text = data.decode('utf-8')
            print(f"Assistant: {greeting_text}")

        # Receive and play audio
        msg_type, data = self.receive_message()
        if msg_type == 5:
            self.play_audio(data)

    def process_voice_input(self):
        """Record audio, send to server, get response"""
        # Record audio
        audio_data = self.record_audio()

        if len(audio_data) < CHUNK_SIZE:
            print("No speech detected. Please try again.")
            return True  # Continue

        # Send audio to server for recognition (type 1)
        print("Sending audio to server...")
        self.send_message(1, audio_data)

        # Receive recognized text (type 2)
        msg_type, data = self.receive_message()
        if msg_type != 2:
            print("Error: Unexpected response from server")
            return True

        recognized_text = data.decode('utf-8')
        if not recognized_text:
            print("No speech detected. Please try again.")
            return True

        print(f"You: {recognized_text}")

        # Send text for LLM processing (type 3)
        self.send_message(3, recognized_text.encode('utf-8'))

        # Receive text response (type 4)
        msg_type, data = self.receive_message()
        if msg_type != 4:
            print("Error: Unexpected response from server")
            return True

        response_text = data.decode('utf-8')
        print(f"Assistant: {response_text}")

        # Receive and play audio response (type 5)
        msg_type, data = self.receive_message()
        if msg_type == 5:
            self.play_audio(data)

        # Check if we should exit
        if any(word in recognized_text.lower() for word in ['exit', 'quit', 'goodbye', 'bye', 'beenden', 'tschüss']):
            return False  # Stop

        return True  # Continue

    def run(self):
        """Main client loop"""
        if not self.connect():
            return

        try:
            # Get greeting
            self.get_greeting()

            # Main interaction loop
            while True:
                if not self.process_voice_input():
                    break

        except KeyboardInterrupt:
            print("\nShutting down...")

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        if self.socket:
            self.socket.close()
        if hasattr(self, 'audio'):
            self.audio.terminate()
        print("Client shut down successfully.")


def main():
    """Entry point"""
    parser = argparse.ArgumentParser(description="Lexi Voice Assistant Client")
    parser.add_argument("--host", default=DEFAULT_SERVER_HOST, help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=DEFAULT_SERVER_PORT, help="Server port (default: 5555)")
    args = parser.parse_args()

    print("=" * 60)
    print("LEXI VOICE ASSISTANT CLIENT")
    print("=" * 60)
    print()

    try:
        client = VoiceAssistantClient(args.host, args.port)
        client.run()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
