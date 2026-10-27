"""Voice loop: listen -> (optionally strip wake phrase) -> ask the LLM -> speak the answer.

Say "quit", "exit", or "stop" to end. Ctrl+C also works.
"""
import sys

from memory_llm import ask
from voice_io import listen, speak

MODE = "raw" if "--raw" in sys.argv[1:] else "hybrid"

WAKE_PHRASES = ["hey roommate", "roommate"]
STOP_WORDS = {"quit", "exit", "stop"}


def strip_wake_phrase(text: str) -> str:
    lowered = text.lower()
    for phrase in WAKE_PHRASES:
        if lowered.startswith(phrase):
            return text[len(phrase):].strip(" ,")
    return text


def main() -> None:
    speak("Roommate is listening. Ask where something was last seen.")
    while True:
        try:
            heard = listen()
        except KeyboardInterrupt:
            speak("Goodbye.")
            break
        if not heard:
            continue
        print(f"Heard: {heard}")
        if heard.strip().lower() in STOP_WORDS:
            speak("Goodbye.")
            break
        question = strip_wake_phrase(heard)
        answer = ask(question, mode=MODE)
        speak(answer)


if __name__ == "__main__":
    main()
