## Raspberry Pi: чек-лист настройки и диагностики аудио

Цель: быстро понять, что **ввод/вывод звука** и уровни работают стабильно.

### 0) Проверка устройств

ALSA:

```bash
arecord -l
aplay -l
arecord -L | head
aplay -L | head
```

PipeWire/PulseAudio (если есть):

```bash
pactl list short sources
pactl list short sinks
```

### 1) Быстрый тест записи микрофона

```bash
arecord -D default -f S16_LE -c1 -r 16000 -d 5 /tmp/mic.wav
aplay /tmp/mic.wav
```

Если слишком тихо/громко — регулируй уровни:

```bash
alsamixer
```

### 2) Быстрый тест вывода

```bash
speaker-test -t sine -f 440 -c 1
```

### 3) Проверка задержки (грубо)

- Если вывод по Bluetooth — задержка часто заметна (100–300+ мс). Для ассистента лучше line-out/USB/I2S.

### 4) Запуск ассистента как service

1) Скопировать проект в `/opt/gpt-station` (как в [`deploy/assistant.service`](deploy/assistant.service:1)).
2) Создать `/etc/assistant.env` по примеру [`deploy/assistant.env.example`](deploy/assistant.env.example:1).

Далее:

```bash
sudo cp deploy/assistant.service /etc/systemd/system/assistant.service
sudo systemctl daemon-reload
sudo systemctl enable assistant
sudo systemctl restart assistant
sudo journalctl -u assistant -f
```

### 5) Режим stop-word

Чтобы работал "стоп/отмена" во время ответа:

- `ENABLE_STOP_WORD=true`
- `SUPPRESS_MIC_DURING_TTS=false`

В корпусе «динамик+микрофон рядом» это может ловить собственный TTS, поэтому:

- держи `STOP_WORDS` коротким,
- при необходимости подними строгость в [`assistant/stop_word.py`](assistant/stop_word.py:1).

