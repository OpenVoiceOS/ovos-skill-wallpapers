"""E2E intent-routing tests for ovos-skill-wallpapers (en-US).

The wallpaper lookups hit a remote image API; they are stubbed so the suite is
deterministic and offline, exercising only intent routing.
"""
from unittest import TestCase
from unittest.mock import patch

from ovos_bus_client.message import Message
from ovos_bus_client.session import Session
from ovoscope import End2EndTest, get_minicroft

SKILL_ID = "skill-ovos-wallpapers.openvoiceos"
LANG = "en-US"

FAKE_WALLPAPERS = ["/tmp/fake_wallpaper_0.jpg", "/tmp/fake_wallpaper_1.jpg"]

# Matches the real default ovos-core pipeline order (padatious/padacioso
# high BEFORE adapt-high) -- see test_golden_utterances.py for why a
# padatious-high-only pipeline hides collisions that only show up once the
# medium-confidence stages get a chance to claim an utterance.
_PIPELINE = [
    "ovos-padatious-pipeline-plugin-high",
    "ovos-padacioso-pipeline-plugin-high",
    "ovos-adapt-pipeline-plugin-high",
    "ovos-padacioso-pipeline-plugin-medium",
    "ovos-adapt-pipeline-plugin-medium",
]


class _IntentRoutingMixin:
    """Shared MiniCroft setup for padacioso intent-routing assertions."""

    @classmethod
    def setUpClass(cls):
        cls._patch = patch(
            "ovos_skill_wallpapers.get_wallpapers",
            return_value=list(FAKE_WALLPAPERS),
        )
        cls._patch.start()
        cls.minicroft = get_minicroft([SKILL_ID])

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "minicroft", None):
            cls.minicroft.stop()
        cls._patch.stop()

    def _assert_routes(self, utterance: str, intent_file: str):
        # dispatched intent messages drop the ".intent" file suffix (OVOS-INTENT-2
        # naming) -- pinning the suffixed form here made the eof_msgs cutoff never
        # fire (so every case ran the full 30s timeout) and the final assertion
        # always fail; see the golden-utterance suite's tolerant matcher for the
        # same fact documented elsewhere in this repo.
        intent_base = intent_file[:-len(".intent")] if intent_file.endswith(".intent") else intent_file
        intent_msg_type = f"{SKILL_ID}:{intent_base}"
        session = Session(f"e2e-en_us-{intent_file}-{hash(utterance)}")
        session.lang = LANG
        session.pipeline = list(_PIPELINE)
        # blacklisted_intents defaults to None on a fresh Session, which
        # crashes the padacioso pipeline (NoneType membership test) - force
        # an empty list.
        session.blacklisted_intents = []
        message = Message(
            "recognizer_loop:utterance",
            {"utterances": [utterance], "lang": LANG},
            {"session": session.serialize()},
        )
        test = End2EndTest(
            minicroft=self.minicroft,
            skill_ids=[SKILL_ID],
            eof_msgs=["ovos.utterance.handled"],
            flip_points=["recognizer_loop:utterance"],
            source_message=message,
            activation_points=[intent_msg_type],
            test_msg_context=False,
            test_message_number=False,
            ignore_messages=[
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
                # intent-service bookkeeping messages that bracket the
                # dispatch (match + handler start/complete); the routing
                # assertion below checks the {SKILL_ID}:{intent} message
                # itself, not these
                "ovos.intent.matched",
                "ovos.intent.handler.start",
                "ovos.intent.handler.complete",
                # wallpaper delivery side effects fired from inside the handler;
                # not relevant to intent-routing assertions
                "ovos.wallpaper.manager.register.provider",
                "ovos.wallpaper.manager.set.wallpaper",
                "ovos.wallpaper.manager.collect.collection.response",
                "homescreen.wallpaper.set",
            ],
            expected_messages=[
                message,
                Message(f"{SKILL_ID}.activate", {}, {"skill_id": SKILL_ID}),
                Message(intent_msg_type, {}, {"skill_id": SKILL_ID}),
                Message("mycroft.skill.handler.start", {}, {"skill_id": SKILL_ID}),
                Message("mycroft.skill.handler.complete", {}, {"skill_id": SKILL_ID}),
                Message("ovos.utterance.handled", {}, {"skill_id": SKILL_ID}),
            ],
        )
        test.execute(timeout=30)


class TestWallpaperRandom(_IntentRoutingMixin, TestCase):
    """Padatious intent: wallpaper_random.intent"""

    def test_change_the_wallpaper(self):
        self._assert_routes(r"change the wallpaper", r"wallpaper_random.intent")

    def test_give_me_a_new_wallpaper(self):
        self._assert_routes(r"give me a new wallpaper", r"wallpaper_random.intent")


class TestPictureRandom(_IntentRoutingMixin, TestCase):
    """Padatious intent: picture_random.intent"""

    def test_show_me_a_random_picture(self):
        self._assert_routes(r"show me a random picture", r"picture_random.intent")


class TestWallpaperAbout(_IntentRoutingMixin, TestCase):
    """Padatious intent: wallpaper_about.intent"""

    def test_set_the_wallpaper_to_space(self):
        self._assert_routes(r"set the wallpaper to space", r"wallpaper_about.intent")


class TestPictureAbout(_IntentRoutingMixin, TestCase):
    """Padatious intent: picture_about.intent"""

    def test_show_me_a_picture_of_cats(self):
        self._assert_routes(r"show me a picture of cats", r"picture_about.intent")
