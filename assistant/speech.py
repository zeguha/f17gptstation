# assistant/speech.py
import queue
import sounddevice as sd
import numpy as np
import platform
import subprocess
import threading
from typing import Optional

audio_queue = queue.Queue()  # сюда кладёт audio_callback numpy-arrays

# Параметры записи
SAMPLE_RATE = 32000   # частота дискретизации для whisper
CHANNELS = 1
CHUNK_SIZE = 480      # ~30 ms при 16k (webrtcvad любит 10/20/30 ms кадры)

# Флаг - говорим ли мы сейчас (потокобезопасен)
SPEAKING_EVENT = threading.Event()

def audio_callback(indata, frames, time, status):
    """
    Callback sounddevice. Помещает моно numpy array (float32 в -1..1) в audio_queue,
    но игнорирует вход, если SPEAKING_EVENT установлен.
    """
    if status:
        # минимальная печать при реальных статусах
        print("audio_callback status:", status, flush=True)
    if SPEAKING_EVENT.is_set():
        return
    arr = indata.copy()
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    arr = arr.astype(np.float32, copy=False)
    audio_queue.put(arr)

# ---------- Синтез речи ----------
class BaseSynth:
    def speak(self, text: str):
        raise NotImplementedError
    def stop(self):
        raise NotImplementedError

class MacNativeSynth(BaseSynth):
    def __init__(self):
        self.proc: Optional[subprocess.Popen] = None
        self.voices = ["Milena", "Anna", "Yuri"]

    def speak(self, text: str):
        self.stop()
        text = text.strip()
        if not text:
            return
        for v in self.voices:
            try:
                self.proc = subprocess.Popen(["say", "-v", v, text])
                self.proc.wait()
                self.proc = None
                return
            except Exception:
                self.proc = None
                continue
        try:
            self.proc = subprocess.Popen(["say", text])
            self.proc.wait()
        except Exception:
            self.proc = None

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.proc = None

class Pyttsx3Synth(BaseSynth):
    def __init__(self):
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 150)
        except Exception:
            self.engine = None

    def speak(self, text: str):
        if not self.engine or not text:
            return
        self.engine.say(text)
        self.engine.runAndWait()

    def stop(self):
        if self.engine:
            try:
                self.engine.stop()
            except Exception:
                pass

class SpeechManager:
    def __init__(self):
        self.synth = None
        self._lock = threading.Lock()
        self._init_synth()

    def _init_synth(self):
        if platform.system() == "Darwin":
            try:
                subprocess.run(["which", "say"], check=True, stdout=subprocess.DEVNULL)
                self.synth = MacNativeSynth()
                return
            except Exception:
                pass
        s = Pyttsx3Synth()
        if s.engine:
            self.synth = s
            return
        self.synth = None

    def speak(self, text: str, async_mode: bool = False):
        if not text or text.strip() == "":
            return
        with self._lock:
            SPEAKING_EVENT.set()
            try:
                if self.synth is None:
                    print(text)
                    SPEAKING_EVENT.clear()
                    return
                if async_mode:
                    t = threading.Thread(target=self._call_synth, args=(text,), daemon=True)
                    t.start()
                else:
                    self._call_synth(text)
            except Exception:
                SPEAKING_EVENT.clear()

    def _call_synth(self, text: str):
        try:
            self.synth.speak(text)
        finally:
            # небольшая пауза после речи, чтобы избежать акустической петли
            threading.Timer(0.25, SPEAKING_EVENT.clear).start()

    def stop(self):
        with self._lock:
            try:
                if self.synth:
                    self.synth.stop()
            except Exception:
                pass
            SPEAKING_EVENT.clear()

# Глобальный менеджер
_speech_manager = SpeechManager()

def speak(text: str):
    _speech_manager.speak(text, async_mode=False)

def speak_async(text: str):
    _speech_manager.speak(text, async_mode=True)

def stop_speech():
    _speech_manager.stop()

# Утилита: выбрать первый доступный вход
def get_default_input_device():
    """Return a stable default input device id.

    On macOS some "virtual" or external devices may be listed first but fail to open
    (AUHAL/PortAudio -9986/-10851). We therefore prefer PortAudio's default input
    device, then fall back to the first usable input device.
    """

    # 1) Prefer PortAudio default input device if available.
    try:
        default_in = sd.default.device[0]
        if default_in is not None and int(default_in) >= 0:
            dev = sd.query_devices(int(default_in), kind="input")
            if dev and dev.get("max_input_channels", 0) > 0:
                return int(default_in)
    except Exception:
        pass

    # 2) Otherwise pick the first input-capable device.
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        try:
            if dev.get("max_input_channels", 0) > 0:
                return idx
        except Exception:
            continue
    raise RuntimeError("Нет доступных входных устройств (микрофонов).")
