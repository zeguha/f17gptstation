from __future__ import annotations


from assistant.weather_skill import _place_candidates_ru


def test_place_candidates_ru_ekaterinburg() -> None:
    c = _place_candidates_ru("екатеринбурге")
    assert "екатеринбурге" in c
    assert "екатеринбург" in c


def test_place_candidates_ru_piter() -> None:
    c = _place_candidates_ru("питере")
    assert "питере" in c
    assert "питер" in c

