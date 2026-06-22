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


if __name__ == "__main__":
    unittest.main()

