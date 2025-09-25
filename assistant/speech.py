import queue
import sounddevice as sd
import numpy as np
import simpleaudio as sa

audio_queue = queue.Queue()

# Параметры записи
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024  # фрейм

def audio_callback(indata, frames, time, status):
    """Добавляет аудио во входную очередь"""
    if status:
        print(status)
    audio_queue.put(indata.copy())

def speak(text):
    """Генерация речи и воспроизведение через gTTS"""
    from gtts import gTTS
    tts = gTTS(text=text, lang='ru')
    tts.save("response.mp3")
    wave_obj = sa.WaveObject.from_wave_file("response.mp3")
    play_obj = wave_obj.play()
    play_obj.wait_done()