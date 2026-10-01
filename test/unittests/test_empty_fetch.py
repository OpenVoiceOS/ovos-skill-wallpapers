"""An empty fetch speaks a dialog and opens no slideshow.

``get_wallpapers`` swallows the request error and returns ``[]`` when the
backend is down (wallhaven.cc answered 521 for a day on 2026-09-19).
``fetch_wallpapers`` indexed ``[0]`` of that, so every handler raised
before ``set_context("SlideShow")`` and the gated next/previous intents
stopped matching for the session, with nothing spoken.

Each of the four fetching handlers is driven with the backend stubbed to
``[]``: it must speak ``no_more_pictures`` (a dialog every shipped locale
carries) and must not set the context. The control runs the same handlers
with one wallpaper and asserts the context IS set.
"""
from unittest.mock import MagicMock, patch

import pytest
from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

import ovos_skill_wallpapers
from ovos_skill_wallpapers import WallpapersSkill

SKILL_ID = "skill-ovos-wallpapers.openvoiceos"
HANDLERS = ["handle_random_wallpaper", "handle_random_picture",
            "handle_wallpaper_about", "handle_picture_about"]


@pytest.fixture
def skill():
    s = WallpapersSkill()
    s._startup(FakeBus(), SKILL_ID)
    s.speak_dialog = MagicMock()
    s.set_context = MagicMock()
    s.change_wallpaper = MagicMock()
    s.gui = MagicMock()
    yield s
    s.default_shutdown()


def _message():
    return Message("test", {"query": "mountains"})


@pytest.mark.parametrize("handler", HANDLERS)
def test_an_empty_fetch_speaks_and_opens_no_slideshow(skill, handler):
    with patch.object(ovos_skill_wallpapers, "get_wallpapers", return_value=[]):
        getattr(skill, handler)(_message())
    spoken = [c.args[0] for c in skill.speak_dialog.call_args_list]
    assert "no_more_pictures" in spoken, spoken
    assert "wallpaper_changed" not in spoken, spoken
    skill.set_context.assert_not_called()


@pytest.mark.parametrize("handler", HANDLERS)
def test_the_control_a_fetch_with_one_wallpaper_opens_the_slideshow(skill, handler):
    with patch.object(ovos_skill_wallpapers, "get_wallpapers", return_value=["/tmp/one.jpg"]):
        getattr(skill, handler)(_message())
    skill.set_context.assert_called_once_with("SlideShow")
    spoken = [c.args[0] for c in skill.speak_dialog.call_args_list]
    assert "no_more_pictures" not in spoken, spoken
