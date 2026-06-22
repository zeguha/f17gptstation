# assistant/main.py
import os
import time
import threading
import tempfile
import subprocess
from collections import deque
import queue
import numpy as np
import soundfile as sf
import re
from difflib import SequenceMatcher

from chat import ask_gpt, ensure_client_ready
from speech import (
    audio_queue,
    audio_callback,
    SPEAKING_EVENT,
    speak,
    speak_async,
    stop_speech,
    SAMPLE_RATE,
    CHANNELS,
    CHUNK_SIZE,
    get_default_input_device,
)

import sounddevice as sd

# Путь к whisper-cli и модели
WHISPER_BIN = os.path.expanduser("/Users/sergejanvarev/gpt-station/whisper.cpp/build/bin/whisper-cli")
MODEL_PATH = os.path.expanduser("/Users/sergejanvarev/gpt-station/whisper.cpp/models/ggml-small.bin")

# Wake/stop
WAKE_WORD = "олег"
STOP_WORD = "стоп"

# Настройки для улучшенной чувствительности при шуме
VAD_MODE = 3
PRE_ROLL_SECONDS = 1.0
POST_SILENCE_SECONDS = 0.8
MAX_RECORD_SECONDS = 15.0

WAKE_TOLERANCE = 0.65

# cooldown после обнаружения wake-фразы
WAKE_COOLDOWN = 1

# Pre-roll buffer
_prebuffer = deque()
_prebuffer_max_chunks = int((PRE_ROLL_SECONDS * SAMPLE_RATE) / CHUNK_SIZE)

# Очереди
recognition_queue = queue.Queue()
results_queue = queue.Queue()

# ignore timestamp
ignore_results_until = 0.0

# Флаг для отслеживания состояния прослушивания
is_processing_command = False

# инициализация Groq клиента заранее для ранней проверки
try:
    ensure_client_ready()
except Exception as e:
    print("Предупреждение: клиент Groq не инициализирован:", e)

# ---- VAD init (webrtcvad optional) ----
try:
    import webrtcvad
    _VAD = webrtcvad.Vad(VAD_MODE)
    HAVE_VAD = True
except Exception:
    _VAD = None
    HAVE_VAD = False

def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^а-яёa-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_wake_word(recognized: str) -> bool:
    if not recognized:
        return False
    got = normalize_text(recognized)
    target = normalize_text(WAKE_WORD)
    if target in got:
        return True
    # fuzzy matching on word windows
    best = 0.0
    words = got.split()
    twords = target.split()
    if not words or not twords:
        return False
    for L in range(max(1, len(twords)-1), len(twords)+2):
        for i in range(0, max(1, len(words)-L+1)):
            frag = " ".join(words[i:i+L])
            r = SequenceMatcher(None, frag, target).ratio()
            if r > best:
                best = r
    return best >= WAKE_TOLERANCE

# ---- recognition worker ----
def float_to_int16_np(arr: np.ndarray) -> np.ndarray:
    clipped = np.clip(arr, -1.0, 1.0)
    pcm16 = (clipped * 32767).astype(np.int16)
    return pcm16

def recognition_worker():
    """Берёт аудио из recognition_queue, вызывает whisper-cli и кладёт результат в results_queue с timestamp."""
    while True:
        item = recognition_queue.get()
        if item is None:
            break
        audio_np = item.get("audio")
        expect_command = item.get("expect_command", False)

        try:
            pcm16 = float_to_int16_np(audio_np)
            tmpf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmpf.close()
            sf.write(tmpf.name, pcm16, SAMPLE_RATE, subtype="PCM_16")
            try:
                proc = subprocess.run(
                    [WHISPER_BIN, "-m", MODEL_PATH, "-f", tmpf.name, "-l", "ru"],
                    capture_output=True,
                    text=True,
                    timeout=40,
                )
                text = proc.stdout.strip().lower()
            except subprocess.TimeoutExpired:
                text = ""
            except Exception as e:
                text = ""
                print("Ошибка выполнения whisper-cli:", e)
        finally:
            try:
                os.unlink(tmpf.name)
            except Exception:
                pass

        # ставим timestamp — время завершения распознавания/обработки
        results_queue.put({"text": text, "expect_command": expect_command, "ts": time.time()})

_rec_thread = threading.Thread(target=recognition_worker, daemon=True)
_rec_thread.start()

# ---- VAD helper ----
def is_speech_frame(frame: np.ndarray) -> bool:
    if HAVE_VAD:
        try:
            pcm16 = ((np.clip(frame, -1, 1) * 32767).astype(np.int16)).tobytes()
            return _VAD.is_speech(pcm16, SAMPLE_RATE)
        except Exception:
            return float(np.abs(frame).mean()) > 0.01
    else:
        return float(np.abs(frame).mean()) > 0.01

# ---- main voice capture with pre-roll & VAD ----
def wait_for_voice():
    buffer_chunks = []
    silence_chunks = 0
    in_speech = False

    while True:
        try:
            frame = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        _prebuffer.append(frame)
        if len(_prebuffer) > _prebuffer_max_chunks:
            _prebuffer.popleft()

        speech = is_speech_frame(frame)

        if speech and not in_speech:
            buffer_chunks = list(_prebuffer)
            buffer_chunks.append(frame)
            in_speech = True
            silence_chunks = 0
            while True:
                try:
                    f = audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                buffer_chunks.append(f)
                if is_speech_frame(f):
                    silence_chunks = 0
                else:
                    silence_chunks += 1
                    if silence_chunks * (CHUNK_SIZE / SAMPLE_RATE) >= POST_SILENCE_SECONDS:
                        break
                total_samples = sum(ch.shape[0] for ch in buffer_chunks)
                if total_samples / SAMPLE_RATE >= MAX_RECORD_SECONDS:
                    break
            audio_np = np.concatenate(buffer_chunks)
            if audio_np.shape[0] / SAMPLE_RATE < 0.25:
                in_speech = False
                buffer_chunks = []
                continue
            return audio_np
        else:
            continue

# ---- main logic with wake-word fuzzy check + ignore window ----
def main():
    global ignore_results_until, is_processing_command

    print("🎤 Ассистент готов. Скажи 'бесполезный автобот' для активации...")

    try:
        device_id = get_default_input_device()
        print("Использован микрофон (device id):", device_id)
    except Exception as e:
        print("❌ Не найден микрофон:", e)
        return

    listening_for_wake = True

    try:
        with sd.InputStream(device=device_id, channels=CHANNELS, callback=audio_callback, samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE):
            while True:
                audio_np = wait_for_voice()
                if audio_np is None:
                    continue

                # отправляем на асинхронное распознавание
                recognition_queue.put({"audio": audio_np, "expect_command": not listening_for_wake})

                # обрабатываем результаты (если есть)
                while not results_queue.empty():
                    item = results_queue.get_nowait()
                    text = item.get("text", "")
                    ts = item.get("ts", 0.0)

                    # Пропускаем результаты, относящиеся к недавно обнаруженному wake
                    if ts <= ignore_results_until:
                        continue

                    # Пропускаем обработку, если уже обрабатываем команду
                    if is_processing_command:
                        continue

                    if listening_for_wake:
                        # проверяем wake-слово
                        if is_wake_word(text):
                            print(f"✅ Wake word распознана: '{text}'")
                            listening_for_wake = False
                            ignore_results_until = time.time() + WAKE_COOLDOWN
                            speak("Слушаю ваш запрос")
                            # НЕ обрабатываем этот текст как команду
                        else:
                            # не wake — игнорируем
                            print(f"👂 Не wake word: '{text}'")
                    else:
                        # мы в режиме ожидания команды
                        if STOP_WORD in text:
                            print("⏹️ Остановка по команде")
                            stop_speech()
                            listening_for_wake = True
                            speak("Отмена")
                        else:
                            # Устанавливаем флаг обработки команды
                            is_processing_command = True
                            
                            print(f"👤 Команда: {text}")
                            speak_async("Обрабатываю")
                            
                            try:
                                answer = ask_gpt(text)
                                print(f"🤖 Ассистент: {answer}")
                                speak(answer)
                            except Exception as e:
                                print(f"❌ Ошибка при обработке команды: {e}")
                                speak("Произошла ошибка при обработке запроса")
                            finally:
                                # Сбрасываем флаг обработки команды
                                is_processing_command = False
                                listening_for_wake = True

                # короткая пауза, чтобы не жечь CPU
                time.sleep(0.03)

    except KeyboardInterrupt:
        print("Выход по Ctrl+C")
    finally:
        recognition_queue.put(None)
        _rec_thread.join(timeout=2)

if __name__ == "__main__":
    main()