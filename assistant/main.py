import subprocess
import numpy as np
import time
from chat import ask_gpt
from speech import audio_callback, audio_queue, speak, SAMPLE_RATE, CHANNELS, CHUNK_SIZE
import sounddevice as sd
import queue
import soundfile as sf

WHISPER_BIN = "../whisper.cpp/build/bin/whisper-cli"
MODEL_PATH = "../models/ggml-model.bin"
WAKE_WORD = "сраный умный"
STOP_WORD = "стоп"
SILENCE_THRESHOLD = 0.01
BUFFER_DURATION = 2  # секунды буфера

def recognize_speech_from_buffer(buffer):
    """Конвертируем аудио буфер в WAV и распознаем через whisper-cli"""
    sf.write("tmp.wav", buffer, SAMPLE_RATE)
    result = subprocess.run([
        WHISPER_BIN,
        "-m", MODEL_PATH,
        "-f", "tmp.wav",
        "-l", "ru"
    ], capture_output=True, text=True)
    return result.stdout.strip().lower()

def main():
    print("🎤 Ассистент готов. Скажи 'Сраный умный' для активации...")

    buffer = []
    max_buffer_len = int(BUFFER_DURATION * SAMPLE_RATE / CHUNK_SIZE)
    listening_for_wake_word = True
    speaking = False

    with sd.InputStream(channels=CHANNELS, callback=audio_callback, samplerate=SAMPLE_RATE):
        while True:
            try:
                data = audio_queue.get(timeout=1)
            except queue.Empty:
                continue

            amplitude = np.abs(data).mean()
            if amplitude > SILENCE_THRESHOLD:
                buffer.append(data)
                if len(buffer) > max_buffer_len:
                    buffer.pop(0)
            else:
                if buffer:
                    audio_data = np.concatenate(buffer)
                    buffer = []

                    text = recognize_speech_from_buffer(audio_data)
                    if not text:
                        continue

                    if listening_for_wake_word:
                        if WAKE_WORD in text:
                            print("🔊 Wake word распознана! Говори команду...")
                            listening_for_wake_word = False
                    else:
                        if STOP_WORD in text:
                            if speaking:
                                print("⏹️ Ответ прерван пользователем.")
                                speaking = False
                            listening_for_wake_word = True
                            continue

                        print("Ты сказал:", text)
                        speaking = True
                        answer = ask_gpt(text)
                        if speaking:
                            print("Ассистент:", answer)
                            speak(answer)
                        speaking = False
                        listening_for_wake_word = True

if __name__ == "__main__":
    main()