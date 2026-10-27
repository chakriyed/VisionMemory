"""Microphone input (speech-to-text) and spoken output (text-to-speech).

STT uses Google's free Web Speech API via SpeechRecognition (needs internet,
no API key). TTS shells out to macOS's built-in `say` command. Swap either
out for a local/offline engine (e.g. Whisper, pyttsx3) later if needed.
"""
import platform
import subprocess

import speech_recognition as sr


def listen(timeout: float = 8, phrase_time_limit: float = 8) -> str:
    """Blocks until a phrase is heard (or timeout), returns the transcript ('' on failure)."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            return ""
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as exc:
        print(f"Speech recognition service error: {exc}")
        return ""


def speak(text: str) -> None:
    print(f"Roommate: {text}")
    if platform.system() == "Darwin":
        subprocess.run(["say", text], check=False)
    else:
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except ImportError:
            print("(no TTS engine available on this platform; install pyttsx3 for speech output)")
