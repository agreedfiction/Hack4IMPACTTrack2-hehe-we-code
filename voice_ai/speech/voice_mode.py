from speech.vosk_listener import listen_once
from core.parser import parse_input
from core.decision_engine import generate_decision
from utils.tts import speak


def run_voice_mode():
    print("🎧 Voice mode started (Ctrl+C to exit)")

    while True:
        try:
            text = listen_once()

            if not text:
                print("⚠️ No speech detected")
                continue

            print("🗣️ You said:", text)

            parsed = parse_input(text)
            decision = generate_decision(parsed)

            print("🤖 Response:", decision)

            speak(decision.replace("₹", "rupees "))

        except KeyboardInterrupt:
            print("\n❌ Voice mode stopped")
            break


if __name__ == "__main__":
    run_voice_mode()