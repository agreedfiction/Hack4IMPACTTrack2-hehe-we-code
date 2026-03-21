from core.parser import parse_input
from core.decision_engine import generate_decision
from utils.tts import speak

def text_mode():
    print("💬 Text mode started (type 'exit' to quit)")

    while True:
        query = input("Enter query: ")

        if query.lower() == "exit":
            break

        print(f"\n👤 You typed: {query}")

        parsed = parse_input(query)
        decision = generate_decision(parsed)

        print(f"🤖 Response: {decision}\n")
        speak(decision.replace("₹", "rupees "))


def voice_mode():
    from speech.vosk_listener import listen_once

    print("🎧 Voice mode started (Ctrl+C to exit)")

    while True:
        try:
            text = listen_once()

            if not text:
                print("⚠️ No speech detected")
                continue

            print(f"\n👤 You said: {text}")

            parsed = parse_input(text)
            decision = generate_decision(parsed)

            print(f"🤖 Response: {decision}\n")
            speak(decision.replace("₹", "rupees "))

        except KeyboardInterrupt:
            print("\n❌ Voice mode stopped")
            break


if __name__ == "__main__":
    print("Select Mode:")
    print("1. Text Mode") 
    print("2. Voice Mode")

    choice = input("Enter choice: ")

    if choice == "1":
        text_mode()
    elif choice == "2":
        voice_mode()
    else:
        print("Invalid choice")