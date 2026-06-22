# assistant/wake_vosk.py
import os
import json
import time
import threading
import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer

MODEL_PATH = os.path.expanduser("~/gpt-station/assistant/models/vosk-model-ru")
SAMPLE_RATE = 16000
BLOCKSIZE = 4000
DEBUG = True

class WakeListener:
    def __init__(self, wake_word: str, callback, model_path: str = MODEL_PATH, sample_rate: int = SAMPLE_RATE):
        if not os.path.isdir(model_path):
            raise RuntimeError(f"Vosk model folder not found: {model_path}")
        self.model = Model(model_path)
        self.wake_word = wake_word.lower()
        grammar = json.dumps([self.wake_word])
        self.recognizer = KaldiRecognizer(self.model, sample_rate, grammar)
        self.sample_rate = sample_rate
        self.callback = callback
        self.stream = None
        self.running = False
        self.thread = None

    def _process_bytes(self, raw_bytes: bytes):
        if not raw_bytes:
            return
        accepted = self.recognizer.AcceptWaveform(raw_bytes)
        if accepted:
            res = json.loads(self.recognizer.Result())
            text = res.get("text", "").strip().lower()
            if DEBUG:
                print("[wake] result:", res)
            if text == self.wake_word:
                threading.Thread(target=self.callback, daemon=True).start()
        else:
            if DEBUG:
                try:
                    p = json.loads(self.recognizer.PartialResult())
                    if p.get("partial"):
                        print("[wake] partial:", p)
                except Exception:
                    pass

    def _sd_callback(self, indata, frames, time_info, status):
        if status and DEBUG:
            print("sounddevice status:", status)
        try:
            raw_bytes = bytes(indata)
        except Exception:
            try:
                arr = np.frombuffer(indata, dtype=np.int16)
                raw_bytes = arr.tobytes()
            except Exception:
                arrf = np.frombuffer(indata, dtype=np.float32)
                arrf = np.nan_to_num(arrf, nan=0.0, posinf=0.0, neginf=0.0)
                arrf = np.clip(arrf, -1.0, 1.0)
                raw_bytes = (arrf * 32767).astype(np.int16).tobytes()
        self._process_bytes(raw_bytes)

    def start(self, device=None):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, args=(device,), daemon=True)
        self.thread.start()

    def _run_loop(self, device):
        try:
            self.stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=BLOCKSIZE,
                device=device,
                dtype='int16',
                channels=1,
                callback=self._sd_callback
            )
            self.stream.start()
            if DEBUG:
                print("WakeListener started (device=%s)" % str(device))
            while self.running:
                time.sleep(0.1)
        except Exception as e:
            print("WakeListener error:", e)
            self.running = False
        finally:
            try:
                if self.stream:
                    self.stream.stop()
                    self.stream.close()
            except Exception:
                pass

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)
            self.thread = None

    def reset(self):
        """Soft reset recognizer without recreating model"""
        self.recognizer.Reset()