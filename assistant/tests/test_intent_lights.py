import unittest

from assistant.intent_lights import detect_lights_intent


class TestLightsIntent(unittest.TestCase):
    def test_power_on_all(self):
        i = detect_lights_intent("везде включи свет")
        self.assertIsNotNone(i)
        assert i is not None
        self.assertEqual(i.action, "power_on")
        self.assertEqual(i.target_scope, "all")

    def test_power_off_room(self):
        i = detect_lights_intent("в детской выключи")
        self.assertIsNotNone(i)
        assert i is not None
        self.assertEqual(i.action, "power_off")
        self.assertEqual(i.target_scope, "room")
        self.assertEqual(i.target_name, "детской")

    def test_brightness_set(self):
        i = detect_lights_intent("приглуши до 30")
        self.assertIsNotNone(i)
        assert i is not None
        self.assertEqual(i.action, "brightness_set")
        self.assertEqual(i.brightness_percent, 30)

    def test_brightness_on_with_percent(self):
        i = detect_lights_intent("везде включи на 50%")
        self.assertIsNotNone(i)
        assert i is not None
        self.assertEqual(i.action, "brightness_set")
        self.assertEqual(i.target_scope, "all")
        self.assertEqual(i.brightness_percent, 50)

    def test_ct_warmer(self):
        i = detect_lights_intent("сделай свет потеплее")
        self.assertIsNotNone(i)
        assert i is not None
        self.assertEqual(i.action, "ct_delta")
        self.assertLess(int(i.ct_delta or 0), 0)

    def test_ct_set_kelvin(self):
        i = detect_lights_intent("в кухне поставь 3000k")
        self.assertIsNotNone(i)
        assert i is not None
        self.assertEqual(i.action, "ct_set")
        self.assertEqual(i.target_scope, "room")
        self.assertEqual(i.ct_kelvin, 3000)

    def test_color_name(self):
        i = detect_lights_intent("поставь синий")
        self.assertIsNotNone(i)
        assert i is not None
        self.assertEqual(i.action, "color_set")
        self.assertIsNotNone(i.color_rgb)

    def test_query_state(self):
        i = detect_lights_intent("как сейчас настроен свет в кухне")
        self.assertIsNotNone(i)
        assert i is not None
        self.assertEqual(i.action, "state_query")

    def _check(self, phrase, action, **fields):
        i = detect_lights_intent(phrase)
        self.assertIsNotNone(i, phrase)
        self.assertEqual(i.action, action, phrase)
        for k, v in fields.items():
            self.assertEqual(getattr(i, k), v, f"{phrase}: {k}")

    def test_relative_brightness_phrasings(self):
        # Real phrasings that previously fell through to the LLM, which then
        # cheerfully answered "сделал" without touching the lamp.
        for p in ["увеличь яркость", "прибавь яркость", "повысь яркость", "сделай поярче",
                  "сделай посветлее", "сделай светлее", "яркость больше", "добавь яркости",
                  "подними яркость"]:
            self._check(p, "brightness_delta", brightness_delta=20)
        for p in ["уменьши яркость", "убавь яркость", "понизь яркость", "снизь яркость",
                  "сделай потемнее", "сделай тусклее", "сделай свет тусклее", "яркость меньше",
                  "приглуши свет"]:
            self._check(p, "brightness_delta", brightness_delta=-20)

    def test_relative_brightness_with_amount(self):
        self._check("убавь яркость на 30", "brightness_delta", brightness_delta=-30)
        self._check("прибавь яркость на 15%", "brightness_delta", brightness_delta=15)
        # ...but an absolute target stays absolute.
        self._check("приглуши до 30", "brightness_set", brightness_percent=30)
        self._check("яркость на 50", "brightness_set", brightness_percent=50)

    def test_brightness_max_min(self):
        self._check("яркость на максимум", "brightness_set", brightness_percent=100)
        self._check("максимальная яркость", "brightness_set", brightness_percent=100)
        self._check("включи свет на полную", "brightness_set", brightness_percent=100)
        self._check("яркость на минимум", "brightness_set", brightness_percent=10)

    def test_music_volume_is_not_lights(self):
        for p in ["убавь громкость", "прибавь громкость", "сделай громче", "сделай тише",
                  "громкость на 50", "перемотай вперед на 30 секунд"]:
            self.assertIsNone(detect_lights_intent(p), p)


if __name__ == "__main__":
    unittest.main()

