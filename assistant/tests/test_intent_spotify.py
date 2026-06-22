import unittest

from assistant.intent_spotify import detect_spotify_intent


class TestSpotifyIntent(unittest.TestCase):
    def test_connect(self):
        i = detect_spotify_intent("подключи spotify")
        self.assertIsNotNone(i)
        assert i is not None
        self.assertEqual(i.action, "spotify_connect")

    def test_seek_forward(self):
        i = detect_spotify_intent("перемотай вперед на 30 секунд")
        self.assertIsNotNone(i)
        assert i is not None
        self.assertEqual(i.action, "seek_rel")
        self.assertEqual(i.seconds, 30)

    def test_seek_back(self):
        i = detect_spotify_intent("перемотай назад на 15 секунд")
        self.assertIsNotNone(i)
        assert i is not None
        self.assertEqual(i.action, "seek_rel")
        self.assertEqual(i.seconds, -15)

    def test_volume(self):
        i = detect_spotify_intent("громкость 55")
        self.assertIsNotNone(i)
        assert i is not None
        self.assertEqual(i.action, "volume")
        self.assertEqual(i.volume_percent, 55)

    def test_play_artist(self):
        i = detect_spotify_intent("включи исполнителя nirvana")
        self.assertIsNotNone(i)
        assert i is not None
        self.assertEqual(i.action, "play_named")
        self.assertEqual(i.content_kind, "artist")
        self.assertEqual(i.query, "nirvana")

    def test_playlist_add_current(self):
        i = detect_spotify_intent("добавь этот трек в плейлист дорога")
        self.assertIsNotNone(i)
        assert i is not None
        self.assertEqual(i.action, "playlist_add")
        self.assertEqual(i.playlist_name, "дорога")
        self.assertIsNone(i.query)

    def test_like_named(self):
        i = detect_spotify_intent("добавь в понравившиеся one more time")
        self.assertIsNotNone(i)
        assert i is not None
        self.assertEqual(i.action, "like_named")


if __name__ == "__main__":
    unittest.main()

