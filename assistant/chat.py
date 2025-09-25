import socket
import socks
import openai
import json
import os
import time
from openai import RateLimitError, APIError, APIConnectionError, Timeout


# Прокси через Xray
socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 1080)
socket.socket = socks.socksocket

# Создай клиент
client = openai.OpenAI(api_key="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")

def ask_gpt(prompt: str, max_retries: int = 5) -> str:
    retry_delay = 1
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except RateLimitError:
            if attempt < max_retries - 1:
                print(f"⚠️ Квота исчерпана, повтор через {retry_delay} сек...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                return "🚨 Квота исчерпана, попробуй позже."
        except (APIConnectionError, Timeout):
            return "⚠️ Проблема с подключением к серверу OpenAI."
        except APIError as e:
            return f"❌ Ошибка API OpenAI: {e}"
        except Exception as e:
            return f"❌ Неизвестная ошибка: {e}"