import sounddevice as sd
import numpy as np
import time

def test_phrase_duration():
    """Тестирует, как долго вы говорите фразу 'сраный умный'"""
    print("🗣️ Произнесите фразу 'сраный умный'...")
    
    recordings = []
    
    def callback(indata, frames, time, status):
        amplitude = np.abs(indata).mean()
        recordings.append((time.inputBufferAdcTime, amplitude))
    
    with sd.InputStream(device=1, channels=1, callback=callback, samplerate=16000):
        input("Нажмите Enter чтобы начать запись на 5 секунд...")
        time.sleep(5)
    
    if recordings:
        start_time = recordings[0][0]
        end_time = recordings[-1][0]
        duration = end_time - start_time
        print(f"📊 Длительность записи: {duration:.2f} секунд")
        
        # Найдем периоды с звуком
        sound_periods = []
        in_sound = False
        sound_start = 0
        
        for timestamp, amplitude in recordings:
            if amplitude > 0.02 and not in_sound:
                in_sound = True
                sound_start = timestamp
            elif amplitude <= 0.02 and in_sound:
                in_sound = False
                sound_periods.append((sound_start, timestamp))
        
        if in_sound:  # если закончили во время звука
            sound_periods.append((sound_start, recordings[-1][0]))
        
        if sound_periods:
            total_sound_duration = sum(end - start for start, end in sound_periods)
            print(f"🔊 Общая длительность звука: {total_sound_duration:.2f} секунд")
            print(f"🔊 Количество звуковых сегментов: {len(sound_periods)}")

if __name__ == "__main__":
    test_phrase_duration()