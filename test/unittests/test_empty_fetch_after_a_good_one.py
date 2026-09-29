"""A failed fetch must not destroy the slideshow that is already on screen.

``fetch_wallpapers`` assigned the backend's result to ``picture_list``
before testing it, so a fetch that returned nothing emptied the list. The
four fetching handlers return early on that and never call ``set_context``,
but a SlideShow context set by an EARLIER good fetch is still live, so the
gated handlers stayed reachable with nothing behind them:
``handle_set_wallpaper`` raised ``IndexError`` and ``handle_next`` set
``pic_idx`` to ``len([]) - 1``, which is -1.

The fix keeps the last good list, so the still-live context stays valid for
what the user can actually see. The gated handlers also refuse an empty list
on their own, because the context can be set by a session this skill did not
open.
"""
from unittest.mock import MagicMock, patch

import pytest
from ovos_utils.fakebus import FakeBus

import ovos_skill_wallpapers
from ovos_skill_wallpapers import WallpapersSkill

SKILL_ID = "skill-ovos-wallpapers.openvoiceos"
GATED = ["handle_set_wallpaper", "handle_next", "handle_prev"]


@pytest.fixture
def skill():
    s = WallpapersSkill()
    s._startup(FakeBus(), SKILL_ID)
    s.speak_dialog = MagicMock()
    s.acknowledge = MagicMock()
    s.change_wallpaper = MagicMock()
    s.gui = MagicMock()
    yield s
    s.default_shutdown()


def _good_then_empty(skill):
    """One good fetch, then one that returns nothing."""
    with patch.object(ovos_skill_wallpapers, "get_wallpapers",
                      return_value=["/tmp/a.jpg", "/tmp/b.jpg"]):
        skill.fetch_wallpapers()
    with patch.object(ovos_skill_wallpapers, "get_wallpapers", return_value=[]):
        assert skill.fetch_wallpapers() is None


def test_an_empty_fetch_keeps_the_last_good_list(skill):
    _good_then_empty(skill)
    assert skill.picture_list == ["/tmp/a.jpg", "/tmp/b.jpg"]


@pytest.mark.parametrize("handler", GATED)
def test_a_gated_handler_survives_an_empty_fetch(skill, handler):
    """Before the fix: handle_set_wallpaper raised IndexError here."""
    _good_then_empty(skill)
    getattr(skill, handler)(None)
    assert skill.pic_idx >= 0, f"{handler} walked pic_idx to {skill.pic_idx}"


def test_handle_next_never_walks_pic_idx_negative_on_an_empty_list(skill):
    """Before the fix: total is 0, so pic_idx became total - 1, which is -1."""
    skill.picture_list = []
    skill.pic_idx = 0
    skill.handle_next(None)
    assert skill.pic_idx >= 0, skill.pic_idx
    assert "no_more_pictures" in [c.args[0] for c in skill.speak_dialog.call_args_list]


@pytest.mark.parametrize("handler", GATED)
def test_a_gated_handler_on_an_empty_list_speaks_instead_of_raising(skill, handler):
    """The context can be set by a session this skill did not open."""
    skill.picture_list = []
    skill.pic_idx = 0
    getattr(skill, handler)(None)
    spoken = [c.args[0] for c in skill.speak_dialog.call_args_list]
    assert "no_more_pictures" in spoken, spoken


def test_the_control_a_good_fetch_still_replaces_the_list(skill):
    """The fix must not freeze the slideshow: a good fetch still wins."""
    with patch.object(ovos_skill_wallpapers, "get_wallpapers",
                      return_value=["/tmp/a.jpg"]):
        skill.fetch_wallpapers()
    with patch.object(ovos_skill_wallpapers, "get_wallpapers",
                      return_value=["/tmp/c.jpg", "/tmp/d.jpg"]):
        assert skill.fetch_wallpapers() == "/tmp/c.jpg"
    assert skill.picture_list == ["/tmp/c.jpg", "/tmp/d.jpg"]
    assert skill.pic_idx == 0
