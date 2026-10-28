"""E2E intent-routing tests for ovos-skill-wallpapers (en-US).

The wallpaper lookups hit a remote image API; they are stubbed so the suite is
deterministic and offline.
"""
from unittest import TestCase
from unittest.mock import patch

from ovos_bus_client.message import Message
from ovos_bus_client.session import Session
from ovoscope import CaptureSession
from ovoscope.routing import PIPELINE, TALLY, get_prototype_minicroft

SKILL_ID = "skill-ovos-wallpapers.openvoiceos"
LANG = "en-US"

FAKE_WALLPAPERS = ["/tmp/fake_wallpaper_0.jpg", "/tmp/fake_wallpaper_1.jpg"]

#: the messages a routed request puts on the bus, in order, once the
#: speech, GUI, context and wallpaper side effects are left out
_IGNORE = {
    "speak",
    "ovos.utterance.speak",
    "recognizer_loop:audio_output_start",
    "recognizer_loop:audio_output_end",
    "mycroft.audio.play_sound",
    "add_context",
    "remove_context",
    "enclosure.mouth.think",
    "enclosure.mouth.reset",
    "enclosure.mouth.viseme_list",
    "gui.clear.namespace",
    "gui.page.show",
    "gui.value.set",
    "mycroft.gui.screen.close",
    "ovos.intent.matched",
    "ovos.intent.handler.start",
    "ovos.intent.handler.complete",
    "ovos.wallpaper.manager.register.provider",
    "ovos.wallpaper.manager.set.wallpaper",
    "ovos.wallpaper.manager.collect.collection.response",
    "homescreen.wallpaper.set",
}


class _IntentRoutingMixin:
    """Shared MiniCroft setup for intent-routing assertions."""

    @classmethod
    def setUpClass(cls):
        cls._patch = patch(
            "ovos_skill_wallpapers.get_wallpapers",
            return_value=list(FAKE_WALLPAPERS),
        )
        cls._patch.start()
        cls.minicroft = get_prototype_minicroft([SKILL_ID], lang=LANG)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "minicroft", None):
            cls.minicroft.stop()
        cls._patch.stop()

    def _assert_routes(self, utterance: str, intent_file: str):
        intent_msg_type = f"{SKILL_ID}:{intent_file.removesuffix('.intent')}"
        session = Session(f"e2e-en_us-{intent_file}-{hash(utterance)}")
        session.lang = LANG
        session.pipeline = list(PIPELINE)
        session.blacklisted_intents = []
        message = Message(
            "recognizer_loop:utterance",
            {"utterances": [utterance], "lang": LANG},
            {"session": session.serialize()},
        )
        capture = CaptureSession(
            self.minicroft,
            eof_msgs=["ovos.utterance.handled", "ovos.intent.unmatched"],
        )
        capture.capture(message, timeout=30)
        types = [m.msg_type for m in capture.finish()]
        TALLY.assert_routed(utterance, intent_msg_type, types)
        self.assertEqual(
            [t for t in types if t not in _IGNORE],
            [
                "recognizer_loop:utterance",
                f"{SKILL_ID}.activate",
                intent_msg_type,
                "mycroft.skill.handler.start",
                "mycroft.skill.handler.complete",
                "ovos.utterance.handled",
            ],
        )


class TestWallpaperRandom(_IntentRoutingMixin, TestCase):
    """wallpaper_random.intent"""

    def test_change_the_wallpaper(self):
        self._assert_routes(r"change the wallpaper", r"wallpaper_random.intent")

    def test_give_me_a_new_wallpaper(self):
        self._assert_routes(r"give me a new wallpaper", r"wallpaper_random.intent")


class TestPictureRandom(_IntentRoutingMixin, TestCase):
    """picture_random.intent"""

    def test_show_me_a_random_picture(self):
        self._assert_routes(r"show me a random picture", r"picture_random.intent")


class TestWallpaperAbout(_IntentRoutingMixin, TestCase):
    """wallpaper_about.intent"""

    def test_set_the_wallpaper_to_space(self):
        self._assert_routes(r"set the wallpaper to space", r"wallpaper_about.intent")


class TestPictureAbout(_IntentRoutingMixin, TestCase):
    """picture_about.intent"""

    def test_show_me_a_picture_of_cats(self):
        self._assert_routes(r"show me a picture of cats", r"picture_about.intent")
