"""End-to-end intent routing tests for the en-US locale.

Each canonical utterance is fired through a real MiniCroft and asserted to
route to the expected Padatious intent. The handlers fetch wallpapers over the
network, so the capture ends at the intent match and asserts routing only.
"""
import unittest

from ovos_bus_client.message import Message
from ovos_bus_client.session import Session
from ovoscope import CaptureSession, get_minicroft

SKILL_ID = "skill-ovos-wallpapers.openvoiceos"
LANG = "en-US"


class TestWallpapersIntentsEnUS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.minicroft = get_minicroft([SKILL_ID])

    @classmethod
    def tearDownClass(cls):
        cls.minicroft.stop()

    def _assert_intent(self, text, intent_file):
        intent_msg = f"{SKILL_ID}:{intent_file}"
        session = Session("test-session")
        session.lang = LANG
        session.pipeline = [
            "ovos-padatious-pipeline-plugin-high",
            "ovos-padatious-pipeline-plugin-medium",
        ]
        utterance = Message(
            "recognizer_loop:utterance",
            {"utterances": [text], "lang": LANG},
            {"session": session.serialize(), "source": "A", "destination": "B"},
        )
        capture = CaptureSession(self.minicroft, eof_msgs=[intent_msg])
        capture.capture(utterance, timeout=30)
        types = [m.msg_type for m in capture.finish()]
        self.assertIn(intent_msg, types)

    def test_change_current_wallpaper(self):
        self._assert_intent("change current wallpaper", "wallpaper.random.intent")

    def test_new_wallpaper(self):
        self._assert_intent("new wallpaper", "wallpaper.random.intent")

    def test_show_me_a_picture(self):
        self._assert_intent("show me a picture", "picture.random.intent")

    def test_display_another_photo(self):
        self._assert_intent("display another photo", "picture.random.intent")

    def test_change_wallpaper_to_nature(self):
        self._assert_intent("change wallpaper to nature", "wallpaper.about.intent")

    def test_new_wall_paper_about_dogs(self):
        self._assert_intent("new wall paper about dogs", "wallpaper.about.intent")

    def test_show_me_a_picture_with_dogs(self):
        self._assert_intent("show me a picture with dogs", "picture.about.intent")

    def test_display_another_image_about_nature(self):
        self._assert_intent("display another image about nature", "picture.about.intent")
