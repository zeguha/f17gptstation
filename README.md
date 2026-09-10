# GPT Station — локальный голосовой ассистент на русском языке

GPT Station — экспериментальный голосовой ассистент для локального компьютера или Raspberry Pi. Проект объединяет wake-word, VAD, ASR, LLM, TTS и детерминированные навыки для погоды, Spotify и умного света.

## Возможности

- Wake-фраза и запись голосовой команды через локальный пайплайн.
- Локальное распознавание через Vosk и опциональный backend whisper.cpp.
- Облачный режим OpenAI для STT, LLM и TTS.
- Погодный навык без LLM на базе Open-Meteo: текущая погода и погода на конкретное время (сегодня/завтра/послезавтра, день недели, утро/день/вечер/ночь) в текущем или названном месте.
- Управление Spotify через OAuth PKCE без хранения пароля.
- Управление WiZ/Gauss лампами по локальной сети.
- Systemd unit для Raspberry Pi.
- Базовый CI для проверки синтаксиса и тестов.

## Требования

- Python 3.11 или новее.
- macOS, Linux, Raspberry Pi OS или Windows (см. заметку ниже про сборку `webrtcvad`).
- Микрофон и устройство вывода звука.
- Для локального Vosk: модель в `assistant/models/vosk-model-ru`.
- Для cloud-режима: OpenAI API key в переменной `OPENAI_API_KEY`.
- Для Spotify: свой `SPOTIFY_CLIENT_ID` из Spotify Developer Dashboard.

## Быстрый старт

macOS / Linux (bash/zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Если PowerShell не даёт выполнить `Activate.ps1` (`running scripts is disabled on this system`), разрешите скрипты для текущего пользователя один раз:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

После активации venv внутри него команда называется `python` (не `python3`) — например: `python -m assistant.spotify_cli auth`.

После копирования заполните `.env` реальными локальными значениями. Файл `.env` игнорируется Git и не должен попадать в репозиторий.

### Заметка про `webrtcvad` на Windows

В `requirements.txt` используется `webrtcvad-wheels` (а не оригинальный `webrtcvad`) — это форк с готовыми wheel-сборками под Windows/macOS/Linux, даёт тот же модуль `webrtcvad`, но не требует компилятора C++ при установке.

Единственное исключение — совсем новые версии Python (на момент написания это Python 3.14): под них готовых wheel'ов ещё нет ни у `webrtcvad`, ни у `webrtcvad-wheels`, и `pip install` попытается собрать из исходников с ошибкой `Microsoft Visual C++ 14.0 or greater is required`. Варианты:

- Использовать Python 3.12 или 3.13 для venv (под них wheel'ы уже есть). Если у вас установлен новый **Python Install Manager** (команда `py`), поставить его можно так: `py install 3.12`, затем `py -3.12 -m venv .venv`.
- Либо поставить [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (компонент "Desktop development with C++") и собрать из исходников.
- Либо пропустить `webrtcvad` вовсе — в коде (`assistant/vad.py`) он опциональный: при отсутствии модуля VAD автоматически откатывается на более грубый энергетический режим определения речи. Достаточно поставить остальные зависимости: `pip install aiohttp numpy requests sounddevice soundfile vosk certifi pytest`.

## Настройка переменных окружения

Основной пример конфигурации находится в `.env.example`. Для Raspberry Pi и systemd используйте `deploy/assistant.env.example` как шаблон для `/etc/assistant.env`.

Минимальная конфигурация cloud-режима:

```bash
OPENAI_API_KEY=replace-with-your-openai-api-key
INPUT_DEVICE=1
WAKE_PHRASE=олег
```

Важные переменные:

| Переменная | Назначение |
| --- | --- |
| `OPENAI_API_KEY` | API-ключ OpenAI. Обязателен для cloud-режима. |
| `OPENAI_BASE_URL` | URL OpenAI-compatible API. По умолчанию `https://api.openai.com/v1`. |
| `OPENAI_STT_MODEL` | Модель распознавания речи. |
| `OPENAI_STT_PROMPT` | Подсказка-словарь для STT, снижает ошибки на доменных словах («плейлист» и т.п.). Можно дописать свои названия плейлистов/исполнителей. |
| `OPENAI_LLM_MODEL` | Модель диалога. |
| `OPENAI_TTS_MODEL` | Модель синтеза речи. |
| `INPUT_DEVICE` | ID микрофона. Часто обязателен на Raspberry Pi. |
| `WAKE_PHRASE` | Wake-фраза ассистента. |
| `VOSK_MODEL_PATH` | Путь к локальной модели Vosk. |
| `SPOTIFY_ENABLED` | Включение Spotify-интеграции. |
| `SPOTIFY_CLIENT_ID` | Client ID вашего Spotify-приложения. |
| `SPOTIFY_TOKEN_PATH` | Локальный путь к OAuth-токенам Spotify. Должен быть в `.secrets/`. |
| `LIGHTS_STATE_PATH` | Локальный state-файл умного света. Должен быть в `.state/`. |
| `WIZ_IPS` | Необязательный список IP-адресов ламп WiZ/Gauss. |

## Локальный запуск

Команды ниже даны для macOS/Linux (`python3`). На Windows, после активации venv (`.venv\Scripts\Activate.ps1`), используйте `python` — команды `python3` в venv нет.

### Cloud-режим

```bash
python3 -m assistant.main_cloud_orchestrated
```

Упрощённый cloud-entrypoint:

```bash
python3 -m assistant.main_cloud
```

### Локальный Vosk-режим

```bash
python3 -m assistant.main_vosk
```

### whisper.cpp backend

```bash
export ASR_BACKEND=whispercpp
export WHISPER_BIN=whisper.cpp/build/bin/whisper-cli
export WHISPER_MODEL=whisper.cpp/models/ggml-small.bin
python3 -m assistant.main_vosk
```

## Spotify

1. Создайте приложение в Spotify Developer Dashboard.
2. Добавьте Redirect URI `http://127.0.0.1:17845/callback`.
3. Укажите в `.env` значения:

```bash
SPOTIFY_ENABLED=true
SPOTIFY_CLIENT_ID=replace-with-your-spotify-client-id
SPOTIFY_REDIRECT_URI=http://127.0.0.1:17845/callback
SPOTIFY_TOKEN_PATH=.secrets/spotify_tokens.json
```

4. Запустите авторизацию:

```bash
python3 -m assistant.spotify_cli auth
```

Токены сохраняются локально в `.secrets/spotify_tokens.json`, этот путь игнорируется Git.

## Умный свет WiZ/Gauss

По умолчанию используется локальный UDP adapter WiZ LAN. Если broadcast не работает, задайте IP ламп вручную:

```bash
WIZ_IPS=192.168.1.10,192.168.1.11
```

State хранится в `.state/lights.json` и не должен публиковаться.

## Запуск через systemd на Raspberry Pi

1. Скопируйте проект в `/opt/gpt-station`.
2. Создайте окружение и установите зависимости.
3. Скопируйте пример env-файла:

```bash
sudo cp deploy/assistant.env.example /etc/assistant.env
sudo nano /etc/assistant.env
```

4. Установите service:

```bash
sudo cp deploy/assistant.service /etc/systemd/system/assistant.service
sudo systemctl daemon-reload
sudo systemctl enable --now assistant.service
```

## Docker

В проекте нет готового Dockerfile для ассистента. Если Docker будет добавлен позже, передавайте секреты через `--env-file .env` или Docker Secrets и не копируйте `.env` внутрь образа.

## Тестирование и проверки

```bash
python -m compileall assistant -x 'assistant/models|__pycache__'
python -m pytest assistant/tests
```

Базовый CI находится в `.github/workflows/ci.yml` и запускает установку зависимостей, проверку синтаксиса и тесты. Секреты в workflow не используются. Если появятся integration-тесты с внешними API, добавляйте ключи только через GitHub Secrets.

## Сборка

Для Python-части отдельный build step не требуется. Для whisper.cpp используйте инструкции upstream-проекта в `whisper.cpp/README.md`; локальные build-артефакты исключены из Git через `.gitignore`.

## Деплой

- Локальный ПК: запускайте нужный entrypoint из активированного virtualenv.
- Raspberry Pi: используйте `deploy/assistant.service` и `/etc/assistant.env`.
- Публичный репозиторий: публикуйте только исходники, примеры конфигурации и документацию; модели, build-артефакты, токены и локальный state не публикуйте.

## Структура проекта

```text
assistant/                 Основной Python-пакет ассистента
assistant/tests/           Unit-тесты
deploy/                    Systemd unit и env-шаблон для Raspberry Pi
whisper.cpp/               Vendored/upstream whisper.cpp для локального ASR backend
.github/workflows/         GitHub Actions CI
.env.example               Безопасный пример локальной конфигурации
requirements.txt           Python-зависимости
SECURITY.md                Правила безопасности
```

## Типичные проблемы

- `OPENAI_API_KEY is not set`: заполните `.env` или переменную окружения.
- `SSLCertVerificationError` на macOS: установите `certifi` из `requirements.txt`.
- Нет активного устройства Spotify: откройте Spotify на нужном устройстве и повторите команду.
- WiZ/Gauss лампы не находятся: проверьте локальное управление в приложении WiZ, одну Wi-Fi сеть и задайте `WIZ_IPS` вручную.
- Vosk model not found: скачайте модель и положите её в `assistant/models/vosk-model-ru` или задайте `VOSK_MODEL_PATH`.

## Безопасность секретов

- Никогда не коммитьте реальные `.env`, `.secrets/`, `.state/`, OAuth-токены, API-ключи, пароли, приватные URL и локальные IP-адреса устройств.
- Используйте `.env.example` и `deploy/assistant.env.example` только с фиктивными значениями.
- В GitHub Actions храните секреты только в GitHub Secrets.
- Если ключ или токен уже попадал в коммит, считайте его скомпрометированным: немедленно отзовите/перевыпустите ключ и очистите историю репозитория с помощью `git filter-repo` или BFG Repo-Cleaner до публикации.
- После очистки истории проверьте репозиторий secret scanner-ом и только затем делайте публичную выгрузку.

## Лицензия и зависимости

Код проекта распространяется по лицензии MIT, см. `LICENSE`. Директория `whisper.cpp/` содержит отдельный upstream-проект со своей лицензией `whisper.cpp/LICENSE`; учитывайте её при распространении.

## Правила для контрибьюторов

1. Делайте изменения в отдельной ветке.
2. Не добавляйте реальные секреты и локальные артефакты.
3. Обновляйте `.env.example`, если добавляете новые переменные окружения.
4. Запускайте тесты перед pull request.
5. Документируйте новые навыки, entrypoint-ы и внешние интеграции.
