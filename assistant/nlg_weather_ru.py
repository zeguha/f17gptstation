"""Russian NLG for the weather skill.

Goal: short, natural phrases.
"""

from __future__ import annotations

from .weather_openmeteo import WeatherNow


def _round_temp_c(x: float) -> int:
    return int(round(float(x)))


def _ms(x: float) -> str:
    # keep it simple for TTS
    v = int(round(float(x)))
    return f"{v} м/с"


def _weather_code_ru(code: int | None) -> str | None:
    # Open-Meteo WMO weather interpretation codes (subset).
    if code is None:
        return None
    m = {
        0: "ясно",
        1: "в основном ясно",
        2: "переменная облачность",
        3: "пасмурно",
        45: "туман",
        48: "изморозь",
        51: "морось",
        53: "морось",
        55: "сильная морось",
        61: "дождь",
        63: "дождь",
        65: "сильный дождь",
        71: "снег",
        73: "снег",
        75: "сильный снег",
        80: "кратковременный дождь",
        81: "ливень",
        82: "сильный ливень",
        95: "гроза",
    }
    return m.get(int(code))


def render_weather_answer_ru(
    w: WeatherNow,
    *,
    place: str | None,
    mode: str,
) -> str:
    t = _round_temp_c(w.temperature_c)
    feels = _round_temp_c(w.apparent_c)
    where = f"в {place}" if place else ""
    desc = _weather_code_ru(w.weather_code)

    if mode == "weather_umbrella":
        p = w.precipitation_prob_pct
        raining = (w.precipitation_mm or 0.0) > 0.0
        if raining or (p is not None and p >= 60):
            if p is not None:
                return f"Лучше возьми зонт: {where} вероятность осадков около {p}%.".strip()
            return f"Лучше возьми зонт: {where} возможны осадки.".strip()
        if p is not None and p >= 30:
            return f"Зонт на всякий случай: вероятность осадков около {p}%.".strip()
        return f"Скорее всего зонт не нужен: {where} в ближайший час осадков не ожидается.".strip()

    if mode == "weather_clothes":
        # base on feels-like
        if feels <= -10:
            tip = "тёплая зимняя куртка, шапка и перчатки"
        elif feels <= 0:
            tip = "тёплая куртка и шапка"
        elif feels <= 10:
            tip = "куртка или ветровка"
        elif feels <= 18:
            tip = "лёгкая куртка или кофта"
        else:
            tip = "лёгкая одежда"
        return f"Сейчас {where} около {t}°, ощущается как {feels}°. Я бы выбрал(а) {tip}.".strip()

    # weather_now (default)
    parts: list[str] = []
    parts.append(f"Сейчас {where} {t}°".strip())
    if feels != t:
        parts.append(f"ощущается как {feels}°")
    if desc:
        parts.append(desc)
    parts.append(f"ветер {_ms(w.wind_ms)}")
    return ". ".join(parts).strip() + "."

