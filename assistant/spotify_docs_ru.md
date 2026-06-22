# Spotify интеграция (голосовое управление)

## Что реализовано в этом коммите

- OAuth **Authorization Code + PKCE** (без хранения пароля)
  - локальный redirect: `http://127.0.0.1:17845/callback`
  - сохранение `refresh_token` и `access_token` в файл с правами `600`
- Асинхронный клиент Spotify Web API с:
  - авто-refresh токена при истечении
  - обработкой `401` (форс-refresh) и `429` (уважаем `Retry-After`)
  - retry/backoff для временных ошибок
- Детерминированный роутинг Spotify-команд до LLM (как погодный скилл)
- Базовые команды:
  - устройства: список, переключение
  - play/pause/stop, next/prev
  - перемотка на `N` секунд вперёд/назад
  - громкость, shuffle, repeat
  - воспроизведение по названию или ссылке (track/album/playlist)
  - очередь (add to queue)
  - like/unlike текущего трека

## Архитектура интеграции

Команды Spotify обрабатываются детерминированно (без LLM), аналогично погоде.

- Нормализация: [`assistant.utils.normalize_text()`](utils.py:11)
- Детекция: [`assistant.intent_spotify.detect_spotify_intent()`](intent_spotify.py:114)
- Исполнение: [`assistant.spotify_skill.handle_spotify_intent()`](spotify_skill.py:59)
- Web API клиент: [`assistant.spotify_client.SpotifyClient`](spotify_client.py:1)
- OAuth/PKCE: [`assistant.spotify_oauth.run_local_pkce_flow()`](spotify_oauth.py:1)
- Хранилище токенов: [`assistant.spotify_tokens.SpotifyTokenStore`](spotify_tokens.py:33)

Интеграция включена в:

- [`assistant/main_cloud_orchestrated.py`](main_cloud_orchestrated.py:1)
- [`assistant/main_cloud.py`](main_cloud.py:1)
- [`assistant/main_vosk.py`](main_vosk.py:1)

## Переменные окружения

См. [`assistant/config.py`](config.py:1).

- `SPOTIFY_ENABLED=true|false`
- `SPOTIFY_CLIENT_ID=your-spotify-client-id` (обязательно для включённой интеграции)
- `SPOTIFY_REDIRECT_URI=http://127.0.0.1:17845/callback`
- `SPOTIFY_TOKEN_PATH=...` (по умолчанию: `./.secrets/spotify_tokens.json`)

## Как подключить аккаунт (OAuth)

1) Создай приложение в Spotify Developer Dashboard.

2) В настройках приложения добавь Redirect URI:

`http://127.0.0.1:17845/callback`

3) Экспортируй `SPOTIFY_CLIENT_ID`:

```bash
export SPOTIFY_CLIENT_ID='your-spotify-client-id'
```

4) Запусти авторизацию:

```bash
python3 -m assistant.spotify_cli auth
```

Скрипт выведет ссылку — открой её в браузере, разреши доступ.

## Если Spotify временно приостановил регистрацию новых приложений

Для Spotify Web API **обязательно нужен** `client_id` из Spotify Developer Dashboard — без него легальный OAuth сделать нельзя.

Что можно сделать:

1) **Проверить, нет ли уже созданного приложения**
   - Dashboard → *Apps* → возможно, у тебя уже есть старое приложение. Тогда просто используй его `client_id`.

2) **Использовать приложение из Team**
   - Если у тебя есть доступ к организации/команде, попроси владельца Team добавить тебя как разработчика и дай `client_id` этого приложения.
   - Затем выставь `SPOTIFY_CLIENT_ID` и добавь Redirect URI `http://127.0.0.1:17845/callback` в настройках приложения.

3) **Подождать восстановления регистрации**
   - Это единственный путь, если нет ни старого приложения, ни доступа к Team.

4) **Временный режим без Spotify API**
   - Можно выставить `SPOTIFY_ENABLED=false` (см. [`assistant/config.py`](config.py:1)) и использовать ассистента без Spotify до появления возможности создать приложение.

Важно: не используйте сторонние/чужие `client_id` и тем более неофициальные методы авторизации — это риск блокировок и нарушение правил Spotify.

## Список требуемых scopes

Полный список в [`assistant/spotify_scopes.py`](spotify_scopes.py:1).

Соответствие команд ↔ scopes:

- play/pause/next/prev/seek/volume/shuffle/repeat/queue/transfer device:
  - `user-modify-playback-state`
  - `user-read-playback-state`
  - `user-read-currently-playing`
- «Понравившиеся»:
  - `user-library-modify`
  - `user-library-read`
- Плейлисты:
  - `playlist-read-private`
  - `playlist-read-collaborative`
  - `playlist-modify-private`
  - `playlist-modify-public`

Минимально для управления воспроизведением:

- `user-modify-playback-state`
- `user-read-playback-state`
- `user-read-currently-playing`

Для лайков:

- `user-library-modify`
- `user-library-read`

Для плейлистов:

- `playlist-modify-private`
- `playlist-modify-public`
- `playlist-read-private`

## Примеры фраз и ожидаемое поведение

- «подключи спотифай» → ассистент объяснит, что нужно запустить CLI авторизацию.
- «какие устройства spotify?» → перечислит доступные устройства.
- «переключи на кухонную колонку» → перенесёт воспроизведение на устройство.
- «перемотай вперёд на 30 секунд» → сделает seek относительно текущей позиции.
- «включи nirvana smells like teen spirit» → найдёт и включит (поиск + play).
- «поставь в очередь daft punk one more time» → добавит найденное в очередь.

## Обработка ошибок

- нет активного устройства: возвращаем короткую инструкцию открыть Spotify/выбрать устройство.
- истёк токен: refresh автоматически.
- 429 rate limit: ждём `Retry-After` и повторяем.
- сеть недоступна/таймаут: ретраи с backoff; затем понятная ошибка.
- 403 нет прав: подсказка переподключить Spotify с нужными scopes.
