import os
import asyncio
import edge_tts
import pyttsx3
import pygame
import socket

# Hide the pygame welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
pygame.mixer.init()

EDGE_VOICE = "en-IN-NeerjaNeural"
OUTPUT_FILE = "output.mp3"

def is_online():
    """Lightning-fast check to see if Wi-Fi/Internet is connected."""
    try:
        # Pings a reliable DNS server. Fails instantly if offline.
        socket.create_connection(("1.1.1.1", 53), timeout=1)
        return True
    except OSError:
        return False

async def gen(text, file_path):
    communicate = edge_tts.Communicate(text, EDGE_VOICE)
    await communicate.save(file_path)

def offline_speak(text):
    """Fallback TTS using pyttsx3 (Strictly English)"""
    print("⚠️ Offline Mode: Using English fallback voice...")
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 160) 
        
        # Lock in an English female voice (like Zira)
        voices = engine.getProperty('voices')
        for voice in voices:
            if "Zira" in voice.name or "female" in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
                
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"❌ Offline TTS failed: {e}")

def speak(text):
    """Smart TTS: Uses Edge if online, instantly uses pyttsx3 if offline."""
    if is_online():
        try:
            asyncio.run(gen(text, OUTPUT_FILE))
            pygame.mixer.music.load(OUTPUT_FILE)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                
            pygame.mixer.music.unload()
            if os.path.exists(OUTPUT_FILE):
                os.remove(OUTPUT_FILE)
        except Exception:
            # If online check passed but Microsoft servers are down
            offline_speak(text)
    else:
        # Instantly trigger offline voice without waiting for DNS errors
        offline_speak(text)