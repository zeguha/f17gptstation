# Голосовой пайплайн (wake → команда → ASR)

Файлы:

- [`assistant/main_vosk.py`](main_vosk.py:1) — основной entrypoint
- [`assistant/pipeline_vosk.py`](pipeline_vosk.py:1) — state machine пайплайна
- [`assistant/audio_stream.py`](audio_stream.py:1) — неблокирующий аудиострим (callback → queue)
- [`assistant/vad.py`](vad.py:1) — VAD (webrtcvad + energy fallback)
- [`assistant/asr_engines.py`](asr_engines.py:1) — единый интерфейс ASR + бэкенды Vosk / whisper.cpp
- [`assistant/postprocess.py`](postprocess.py:1) — нормализация + страховочная отсечка wake-фразы

## Запуск

1) Установить зависимости:

```bash
python3 -m pip install -U sounddevice soundfile numpy vosk requests
python3 -m pip install -U webrtcvad  # опционально, но рекомендуется
python3 -m pip install -U "setuptools<81"  # опционально: глушит warning про pkg_resources
```

2) Экспортировать ключ (если используете LLM через Groq):

```bash
export GROQ_API_KEY='replace-with-your-groq-api-key'
```

Примечания по предупреждениям:

- `webrtcvad`: warning про `pkg_resources` безопасен. Его можно игнорировать, либо поставить `setuptools<81`.
- Vosk: warning `Runtime graphs are not supported by this model` означает, что runtime-grammar отключена в модели.
  В пайплайне wake определяется через fuzzy prefix matching, поэтому это предупреждение не ломает работу.

3) Запустить:

```bash
python3 -m assistant.main_vosk
```

## Cloud-режим (рекомендуется для Raspberry Pi 3B)

Локально остаются wake/VAD и запись, а распознавание/ответ/голос — в облаке.

Entry points:

- простой: [`assistant/main_cloud.py`](main_cloud.py:1)
- production (очереди + stop-word): [`assistant/main_cloud_orchestrated.py`](main_cloud_orchestrated.py:1)

Минимально:
```bash
export OPENAI_API_KEY='...'
export INPUT_DEVICE=1
export WAKE_PHRASE='олег'
python3 -m assistant.main_cloud_orchestrated
```

systemd unit: [`deploy/assistant.service`](../deploy/assistant.service:1)

Пример env-файла для systemd: [`deploy/assistant.env.example`](../deploy/assistant.env.example:1)

По умолчанию:

- wake/VAD/запись команды — локально (Vosk+VAD)
- команда распознаётся в облаке (STT)
- ответ генерируется LLM
- озвучка — TTS

### Stop-word (barge-in)

В orchestrated entrypoint есть best-effort остановка проигрывания по словам `стоп/хватит/отмена`.

Важно:

- если включён `SUPPRESS_MIC_DURING_TTS=true`, микрофон будет глушиться во время TTS — stop-word работать не сможет.
- для stop-word выставь `SUPPRESS_MIC_DURING_TTS=false` и `ENABLE_STOP_WORD=true` (см. [`deploy/assistant.env.example`](../deploy/assistant.env.example:1)).

Это работает без AEC, поэтому детектор сделан очень строгим.

---

## macOS: ошибка SSL CERTIFICATE_VERIFY_FAILED

Если при обращении к OpenAI получаете:

`SSLCertVerificationError: certificate verify failed: unable to get local issuer certificate`

то это типично для python.org Python на macOS без установленных CA.

Решение (рекомендуемое):

```bash
python3 -m pip install -U certifi
```

В коде OpenAI клиента добавлена поддержка `certifi` (см. [`assistant/cloud_openai.py`](cloud_openai.py:1)).

---

## Ошибка OpenAI HTTP 400 про temperature

Некоторые модели не поддерживают кастомный `temperature` (разрешён только дефолт).

Если видишь:
`Unsupported value: 'temperature' ... Only the default (1) value is supported.`

то выставь:

```bash
export OPENAI_LLM_TEMPERATURE=default
```

или удали переменную совсем (тогда используется дефолт из [`assistant/config.py`](config.py:32)).

---

## Ошибка OpenAI TTS HTTP 404: Invalid URL (POST /v1/audio/speech)

Обычно это значит, что выбранная модель **не поддерживает** HTTPS endpoint `/audio/speech`.
Частый пример: realtime-модели.

Решение:

- используй TTS модель, которая поддерживает `/audio/speech`, например:
  - `OPENAI_TTS_MODEL=gpt-4o-mini-tts`
- или верни дефолт из [`assistant/config.py`](config.py:33).

---

## Калибровка wake порогов

Скрипт собирает статистику по реальным срабатываниям и предлагает стартовые значения:

```bash
python3 -m assistant.calibrate_wake --seconds 90 --log INFO
```

Используй предложенные `WAKE_MATCH_THRESHOLD` и `WAKE_MIN_SPEECH_RATIO`.

## Выбор ASR бэкенда для команды

### Vosk (по умолчанию)

```bash
export ASR_BACKEND=vosk
python3 -m assistant.main_vosk
```

### whisper.cpp (локально, качество выше, но медленнее)

```bash
export ASR_BACKEND=whispercpp
export WHISPER_BIN='whisper.cpp/build/bin/whisper-cli'
export WHISPER_MODEL='whisper.cpp/models/ggml-small.bin'
python3 -m assistant.main_vosk
```

## Настройки, которые можно крутить (env)

- `WAKE_PHRASE` — wake-фраза
- `WAKE_MATCH_THRESHOLD` — порог fuzzy-совпадения wake (для однословных wake обычно 0.5–0.7)
- `WAKE_MIN_SPEECH_RATIO` — сколько "speech" должно быть по VAD в окне подтверждения wake (если wake «не срабатывает» — ставьте 0.0–0.1)
- `SAMPLE_RATE` — частота дискретизации (рекомендуется 16000)
- `VAD_MODE` — webrtcvad режим 0..3 (для шумной среды 2–3)
- `ASSISTANT_LOG_LEVEL` — DEBUG/INFO
- `VOSK_MODEL_PATH` — путь к модели Vosk
- `INPUT_DEVICE` — id устройства ввода (полезно на macOS, если первый девайс не открывается)

### Если говорите быстро (wake + вопрос слитно)

Если произнести «олег как у тебя дела» без паузы, Vosk часто финализирует это как один кусок.
Раньше такой кусок воспринимался как wake и команда терялась.

Теперь пайплайн извлекает команду прямо из этого результата: см. [`WakeCommandPipeline.wait_for_wake()`](pipeline_vosk.py:136) и `StopReason.EMBEDDED_IN_WAKE`.

Рекомендации для реального микрофона:

- `SAMPLE_RATE=16000`, `frame_ms=30` (в коде) — совместимо с webrtcvad, стабильная задержка
- `VAD_MODE=2` для дома / `VAD_MODE=3` для шумного офиса
- `wake_tail_drop_ms=250` (в коде) — чтобы гарантированно отсечь «хвост» wake-фразы
