import sounddevice as sd
import queue
import json
import time
import keyboard
import numpy as np
from vosk import Model, KaldiRecognizer

# 1. LOAD THE HEAVY MODEL GLOBALLY
print("⏳ Loading Vosk AI Model (1GB)... This may take 10-15 seconds...")
try:
    # Updated to the new 1GB model folder name
    model = Model("models/vosk-model-en-in-0.5")
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Failed to load model. Check your path.", e)

q = queue.Queue()

# ==========================================
# 🎚️ BACKGROUND NOISE THRESHOLD
# ==========================================
# Set to 400 for a loud room. Adjust up if it catches fans, down if it cuts your voice.
NOISE_THRESHOLD = 500
# ==========================================

def callback(indata, frames, time_info, status):
    """Calculates volume. Keeps loud sounds, mutes quiet background noise."""
    # 1. Convert raw bytes to a mathematical array
    audio_data = np.frombuffer(indata, dtype=np.int16)
    
    # 2. Calculate volume (convert to float32 to prevent math overflow)
    volume = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
    
    if volume > NOISE_THRESHOLD:
        # It's loud enough, keep the audio
        q.put(bytes(indata))
    else:
        # It's background noise, send pure silence (zero-bytes) to Vosk
        q.put(b'\x00' * len(indata))

def listen_once():
    try:
        # Create a fresh recognizer for every new sentence
        rec = KaldiRecognizer(model, 16000)

        # Empty any old noise from the queue
        while not q.empty():
            q.get_nowait()

        print("\n🎤 HOLD 'U' to speak...")

        with sd.RawInputStream(
            samplerate=16000,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=callback,
        ):
            # Phase 1: Idle - Wait for 'U' key
            while not keyboard.is_pressed('u'):
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                time.sleep(0.01)

            print("🎙️ Listening... (Speak naturally!)")

            # Phase 2: Active Capture
            while keyboard.is_pressed('u'):
                try:
                    data = q.get(timeout=0.1)
                    rec.AcceptWaveform(data)
                except queue.Empty:
                    continue

            # Phase 3: The Tail Buffer (0.6 seconds safety net)
            timeout_time = time.time() + 0.6
            while time.time() < timeout_time:
                try:
                    data = q.get(timeout=0.1)
                    rec.AcceptWaveform(data)
                except queue.Empty:
                    continue

        print("⏹️ Processing...")

        # Fetch the final text
        res = json.loads(rec.FinalResult())
        result_text = res.get("text", "")

        return result_text.strip()

    except Exception as e:
        print("❌ Audio Error:", e)
        return ""