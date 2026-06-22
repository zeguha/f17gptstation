# assistant/voice_vosk.py
import os
import json
import time
import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer

MODEL_PATH = os.path.expanduser("~/gpt-station/assistant/models/vosk-model-ru")
SAMPLE_RATE = 16000
BLOCKSIZE = 4000
DEBUG = False

class SpeechRecognizer:
    def __init__(self, model_path: str = MODEL_PATH, sample_rate: int = SAMPLE_RATE):
        if not os.path.isdir(model_path):
            raise RuntimeError(f"Vosk model folder not found: {model_path}")
        self.model = Model(model_path)
        self.sample_rate = sample_rate
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.stream = None

    def _coerce_to_bytes(self, data):
        try:
            return bytes(data)
        except Exception:
            pass
        try:
            arr = np.frombuffer(data, dtype=np.int16)
            return arr.tobytes()
        except Exception:
            pass
        try:
            arrf = np.frombuffer(data, dtype=np.float32)
            arrf = np.nan_to_num(arrf, nan=0.0, posinf=0.0, neginf=0.0)
            arrf = np.clip(arrf, -1.0, 1.0)
            return (arrf * 32767).astype(np.int16).tobytes()
        except Exception:
            arr_any = np.asarray(data)
            if arr_any.dtype in [np.float32, np.float64]:
                arrf = np.nan_to_num(arr_any.astype(np.float32))
                arrf = np.clip(arrf, -1.0, 1.0)
                return (arrf * 32767).astype(np.int16).tobytes()
            elif np.issubdtype(arr_any.dtype, np.integer):
                return arr_any.astype(np.int16).tobytes()
        return b""

    def listen(self, timeout=10, min_words=1):
        text_parts = []
        last_speech_time = time.time()
        end_silence_timeout = 0.8
        start_time = time.time()

        try:
            with sd.RawInputStream(samplerate=self.sample_rate, blocksize=BLOCKSIZE,
                                dtype='int16', channels=1) as stream:
                while True:
                    try:
                        data, overflow = stream.read(BLOCKSIZE)
                    except Exception:
                        time.sleep(0.01)
                        continue

                    raw_bytes = self._coerce_to_bytes(data)
                    if not raw_bytes:
                        continue

                    arr_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
                    amplitude = float(np.abs(arr_int16).mean())
                    if amplitude > 150:
                        last_speech_time = time.time()

                    if self.recognizer.AcceptWaveform(raw_bytes):
                        res = json.loads(self.recognizer.Result())
                        t = res.get("text", "").strip().lower()
                        if t:
                            text_parts.append(t)
                            last_speech_time = time.time()
                    # partial ignored

                    if text_parts and (time.time() - last_speech_time) > end_silence_timeout:
                        break
                    if (time.time() - start_time) > timeout:
                        break

        except Exception:
            return ""

        final = " ".join(text_parts).strip().lower()
        if len(final.split()) < min_words:
            return ""
        return final

    def reset(self):
        """Soft reset recognizer"""
        self.recognizer.Reset()