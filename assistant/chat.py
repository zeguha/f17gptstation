# assistant/chat.py
import os
import time
import requests

from .env import load_env_file


load_env_file()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Предпочтительный порядок моделей (если они есть в списке от Groq, будем выбирать по порядку)
PREFERRED_MODEL_IDS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "groq/compound-mini",
    "groq/compound"
]

# Локальный запасной список (если запрос к Groq API не удался)
FALLBACK_MODELS = [
    "groq/compound-mini",
    "groq/compound"
]

MODELS_ENDPOINT = "https://api.groq.com/openai/v1/models"

class GroqClientWrapper:
    def __init__(self, api_key: str):
        if not api_key:
            raise RuntimeError("GROQ_API_KEY не задан в окружении")
        self.api_key = api_key
        self._client = None  # ленивый импорт groq lib может быть сделан по желанию

    def list_models(self):
        """Запрашивает у Groq актуальный список моделей (возвращает list of model id strings)."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.get(MODELS_ENDPOINT, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            # ответ совместим с OpenAI Models endpoint — ищем 'data' или прямой список
            model_ids = []
            if isinstance(data, dict) and "data" in data:
                for item in data["data"]:
                    # item может быть словарём с 'id'
                    if isinstance(item, dict) and "id" in item:
                        model_ids.append(item["id"])
                    elif isinstance(item, str):
                        model_ids.append(item)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "id" in item:
                        model_ids.append(item["id"])
                    elif isinstance(item, str):
                        model_ids.append(item)
            return model_ids
        except Exception as e:
            # не фатальная ошибка — вернём пустой список, чтобы код мог использовать FALLBACK_MODELS
            print("⚠️ Не удалось получить список моделей от Groq API:", e)
            return []

    def create_chat_completion(self, model: str, messages: list, **kwargs):
        """
        Создаёт чат-запрос используя groq-клиент (если доступен) или через HTTP.
        Тут мы делаем совместимый с OpenAI POST /openai/v1/chat/completions.
        """
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {"model": model, "messages": messages}
        body.update(kwargs)
        resp = requests.post(url, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        return resp.json()

# Глобальные объекты
_groq_wrapper = None
_available_models = []

def _init_groq_wrapper():
    global _groq_wrapper, _available_models
    if _groq_wrapper is not None:
        return
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY не найден в окружении")
    _groq_wrapper = GroqClientWrapper(GROQ_API_KEY)
    # попробуем получить список моделей у Groq
    remote_models = _groq_wrapper.list_models()
    if remote_models:
        # используем только уникальные ids
        _available_models = list(dict.fromkeys(remote_models))
        print("✅ Получены модели от Groq API — выбрано", len(_available_models))
    else:
        # Если не удалось получить через API — используем fallback локальный список
        _available_models = FALLBACK_MODELS[:]
        print("⚠️ Используются запасные модели:", _available_models)

def _pick_model():
    """Выбирает лучшую модель из доступных (предпочтение по PREFERRED_MODEL_IDS)."""
    if not _available_models:
        _init_groq_wrapper()
    # попытка выбрать по списку предпочтений
    for pref in PREFERRED_MODEL_IDS:
        if pref in _available_models:
            return pref
    # иначе — возвращаем просто первую доступную
    if _available_models:
        return _available_models[0]
    # совсем нет моделей — бросаем ошибку
    raise RuntimeError("Нет доступных моделей (Groq). Проверь GROQ_API_KEY или доступность сервиса.")

def ask_gpt(prompt: str, max_retries: int = 3) -> str:
    """Обёртка: вызывает Groq chat completion и возвращает текст ответа."""
    _init_groq_wrapper()
    model = _pick_model()
    messages = [{"role": "user", "content": prompt}]
    last_err = None

    for attempt in range(max_retries):
        try:
            resp = _groq_wrapper.create_chat_completion(model=model, messages=messages, temperature=0.7, max_tokens=1024)
            # ожидаем структуру OpenAI-like: resp['choices'][0]['message']['content']
            if isinstance(resp, dict):
                choices = resp.get("choices") or []
                if choices and isinstance(choices, list):
                    first = choices[0]
                    msg = first.get("message") or first.get("text") or {}
                    if isinstance(msg, dict):
                        return msg.get("content", "").strip()
                    elif isinstance(first.get("text", ""), str):
                        return first.get("text", "").strip()
            return str(resp)
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            print("Ошибка API:", e)
            # Если модель помечена decommissioned — удаляем её из списка и пробуем следующую
            if "decommission" in err_str or "model_decommissioned" in err_str:
                print(f"⚠️ Модель {model} помечена как decommissioned — убираю из пула.")
                if model in _available_models:
                    _available_models.remove(model)
                # попытаемся выбрать новую модель (если есть)
                try:
                    model = _pick_model()
                    print("Пробую модель:", model)
                except Exception as ex:
                    print("❌ Нет доступных моделей после удаления:", ex)
                    break
                continue
            # для прочих ошибок делаем экспоненциальный бэкофф
            sleep_for = 2 ** attempt
            time.sleep(sleep_for)
    return f"Ошибка при запросе к модели: {last_err}"

# Вспомогательная функция для ручного теста/отладки
def list_available_models():
    _init_groq_wrapper()
    return _available_models[:]

# --- Backward-compatible helper (вставить в конец файла chat.py) ---
def ensure_client_ready():
    """
    Совместимая с прошлой версией функция.
    Инициализирует обёртку Groq и набор доступных моделей.
    Бросает RuntimeError при проблемах (как раньше).
    """
    try:
        _init_groq_wrapper()
        return True
    except Exception as e:
        raise RuntimeError(f"Не удалось инициализировать Groq client: {e}")
