"""Real bus round-trip coverage for the ``{query}`` slot blacklist
(``locale/en-US/query.blacklist``) and its handler-side fallback.

``ovos_workshop.skills.ovos.OVOSSkill.register_intent_file`` (OVOS-INTENT-2
§4.3) reads every ``{slot}`` in a padatious ``.intent`` template and, for
each one, loads a sibling ``<slot>.blacklist`` file of values that must not
bind to that slot. ``wallpaper_about.intent`` and ``picture_about.intent``
both declare ``{query}``, and ``query.blacklist`` lists anaphoric fillers
("that", "this", "it", ...) so a bare deictic reference does not get treated
as a search topic.

A blacklisted slot value does not stop the intent from matching: the slot
is treated as unresolved (OVOS-INTENT-2 §4.3) and the intent still fires
with ``query`` missing from ``match_data``.

The pipeline receives ``query.blacklist`` with the intent registration, and
``_resolve_query`` in ``__init__.py`` re-checks the bound value against the
same file in the handler.

This suite fires real utterances through a ``MiniCroft`` and asserts on
what the skill actually searched for (via the ``get_wallpapers`` call the
handler makes), not on the bus message types alone.
"""
from unittest.mock import patch

import pytest
from ovos_bus_client.message import Message
from ovos_bus_client.session import Session
from ovoscope import CaptureSession
from ovoscope.routing import PIPELINE, TALLY, get_prototype_minicroft

SKILL_ID = "skill-ovos-wallpapers.openvoiceos"
LANG = "en-US"

FAKE_WALLPAPERS = ["/tmp/fake_wallpaper_0.jpg", "/tmp/fake_wallpaper_1.jpg"]


@pytest.fixture(scope="module")
def minicroft():
    with patch("ovos_skill_wallpapers.get_wallpapers", return_value=list(FAKE_WALLPAPERS)) as mocked:
        mc = get_prototype_minicroft([SKILL_ID], lang=LANG)
        yield mc, mocked
        mc.stop()


def _fire(mc, mocked, text):
    mocked.reset_mock()
    session = Session(f"blacklist-{text}")
    session.lang = LANG
    session.pipeline = list(PIPELINE)
    utterance = Message(
        "recognizer_loop:utterance",
        {"utterances": [text], "lang": LANG},
        {"session": session.serialize(), "source": "A", "destination": "B"},
    )
    capture = CaptureSession(
        mc, eof_msgs=["ovos.utterance.handled", "ovos.intent.unmatched"]
    )
    capture.capture(utterance, timeout=30)
    types = [m.msg_type for m in capture.finish()]
    # positional or keyword "query" arg the skill actually searched for
    query = None
    if mocked.call_args is not None:
        args, kwargs = mocked.call_args
        query = kwargs.get("query", args[0] if args else None)
    return types, query


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "text",
    [
        "set the wallpaper to that",
        "set the wallpaper to this",
        "change the wallpaper to that one",
        "change the wallpaper to the same",
        "show me a picture of it",
    ],
)
def test_deictic_query_is_not_claimed_as_topic(minicroft, text):
    """A bare deictic reference must never be searched for as if it were a
    real topic, whichever intent claims it. The intent can still fire --
    the handler must never crash and must never hand the deictic word
    itself to ``get_wallpapers``."""
    mc, mocked = minicroft
    types, query = _fire(mc, mocked, text)
    assert "mycroft.skill.handler.error" not in types, (
        f"{text!r} crashed the handler: {types!r}"
    )
    assert query is None, (
        f"{text!r} was searched for as a literal topic: "
        f"get_wallpapers(query={query!r})"
    )


@pytest.mark.timeout(60)
def test_real_topic_query_is_claimed(minicroft):
    """Sanity check: a real topic in the same slot position still matches
    and is actually searched for, so the blacklist/fallback path is
    filtering the deictic fillers specifically, not breaking the intent."""
    mc, mocked = minicroft
    text = "set the wallpaper to space"
    types, query = _fire(mc, mocked, text)
    TALLY.assert_routed(text, f"{SKILL_ID}:wallpaper_about", types)
    assert query == "space", (
        f"expected get_wallpapers(query='space'), got {query!r}"
    )


@pytest.mark.timeout(60)
def test_deictic_phrase_with_real_topic_is_claimed(minicroft):
    """Both the slot blacklist and the ``_resolve_query`` fallback match by
    whole-value equality, not substring -- "that beach" is not "that", so a
    deictic word merely appearing inside a longer, genuine topic must still
    bind and be searched for verbatim."""
    mc, mocked = minicroft
    text = "set the wallpaper to that beach"
    types, query = _fire(mc, mocked, text)
    TALLY.assert_routed(text, f"{SKILL_ID}:wallpaper_about", types)
    assert query == "that beach", (
        f"expected get_wallpapers(query='that beach'), got {query!r}"
    )
