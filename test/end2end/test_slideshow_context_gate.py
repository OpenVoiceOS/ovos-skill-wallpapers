"""Two-turn coverage for the "SlideShow" context gate on
``next_picture.intent``/``previous_picture.intent``/``make_wallpaper.intent``.

Those three intents are ``.intent`` files gated via
``@intent_handler(..., requires_context=["SlideShow"])`` (see
``__init__.py``). ``self.set_context("SlideShow")`` is only
set by the user-facing handlers that actually start a slideshow or show a
picture/wallpaper (``handle_random_wallpaper``, ``handle_random_picture``,
``handle_wallpaper_about``, ``handle_picture_about``), so the gate should
only open after the user has done that in the current session.

This module asserts both directions of that behavior for each of the three
gated intents:

* POSITIVE: prime a session with "show me a picture" (opens the gate), then
  fire the intent's canonical utterance in the SAME session -- should match.
* NEGATIVE: fire the same utterance in a brand-new, unprimed session --
  should NOT match (this is the direction that was a permanent false
  positive before the boot-time context leak documented below was fixed).

``_fire()`` captures the session off the orchestrator's ``ovos.utterance.handled``
end-marker (OVOS-PIPELINE-1 §9.5) instead of reading the orchestrator's
private ``SessionManager.sessions`` registry: that registry is
default-session-only per spec and never holds a named conversation session's
real state. As of the SESSION-2 §2.6 completion sync (ovos-core 3.2.5a1),
the handler's session write is folded into the round's working session and
re-stamped on that end-marker's ``context["session"]`` before it fires, so
``ovos.utterance.handled`` always reflects the context the handler set,
regardless of whether the handler speaks before or after writing it
(``handle_random_picture`` speaks THEN calls ``self.set_context(...)``).
The captured session is re-declared on the follow-up turn's utterance,
exactly as a real client would.

Version discrimination for the SESSION-2 §2.6 completion sync itself is
proven by ovos-core#921's own 4-process bus rig, not by this module: ovoscope
delivers the same in-process ``Message`` object to every bus subscriber, so a
skill's ``SessionManager.get(message)`` binds straight to the orchestrator's
own working-session object and the write lands without the fold. A real bus
deserializes each subscriber's own copy, which is what the fold exists for.
Until ovoscope's in-process delivery gets its own serialization fix, running
this module against ovos-core 3.2.4a2 and 3.2.5a1 looks identical here.
"""
from unittest.mock import patch

import pytest
from ovos_bus_client.message import Message
from ovos_bus_client.session import Session
from ovoscope import CaptureSession
from ovoscope.routing import PIPELINE, TALLY, get_prototype_minicroft

SKILL_ID = "skill-ovos-wallpapers.openvoiceos"
LANG = "en-US"

_IGNORE = [
    "speak",
    "ovos.utterance.speak",
    "mycroft.audio.play_sound",
    "enclosure.mouth.text",
    "enclosure.mouth.reset",
    "enclosure.mouth.events.deactivate",
    "enclosure.mouth.events.activate",
]

# one (utterance, intent_label) pair per migrated intent, drawn verbatim
# from that intent's own .intent grammar so it is guaranteed to match once
# (and only once) the "SlideShow" context is open.
_GATED_CASES = [
    ("next picture", "next_picture"),
    ("previous picture", "previous_picture"),
    ("set this as the wallpaper", "make_wallpaper"),
]


#: The priming intent calls ``fetch_wallpapers``, which reaches wallhaven.cc
#: unless the backend is stubbed. On 2026-09-19 wallhaven answered 521 for
#: a day, the fetch came back empty, and every case here failed for a
#: reason that had nothing to do with the gate. The stub keeps this test
#: about the gate.
FAKE_WALLPAPERS = ["/tmp/fake_wallpaper_0.jpg", "/tmp/fake_wallpaper_1.jpg"]


@pytest.fixture(scope="module")
def minicroft():
    with patch("ovos_skill_wallpapers.get_wallpapers", return_value=list(FAKE_WALLPAPERS)):
        mc = get_prototype_minicroft([SKILL_ID], lang=LANG)
        yield mc
        mc.stop()


def _session(session_id):
    session = Session(session_id)
    session.lang = LANG
    session.pipeline = list(PIPELINE)
    session.blacklisted_intents = []
    return session


def _fire(mc, session, text):
    utterance = Message(
        "recognizer_loop:utterance",
        {"utterances": [text], "lang": LANG},
        {"session": session.serialize(), "source": "A", "destination": "B"},
    )
    capture = CaptureSession(
        mc,
        eof_msgs=["ovos.utterance.handled", "ovos.intent.unmatched"],
        ignore_messages=_IGNORE,
    )
    capture.capture(utterance, timeout=30)
    messages = capture.finish()
    types = [m.msg_type for m in messages]
    carried_session = session
    for m in messages:
        if m.msg_type == "ovos.utterance.handled" and m.context.get("session"):
            carried_session = Session.deserialize(m.context["session"])
    return types, carried_session


@pytest.mark.timeout(90)
@pytest.mark.parametrize("utterance,intent_label", _GATED_CASES, ids=[c[1] for c in _GATED_CASES])
def test_gated_intent_matches_after_priming_slideshow(minicroft, utterance, intent_label):
    """Starting a slideshow in a session should open the gate for the
    migrated intent within that SAME session."""
    session = _session(f"gate-positive-{intent_label}")
    prime_types, session = _fire(minicroft, session, "show me a picture")
    TALLY.assert_routed("show me a picture", f"{SKILL_ID}:picture_random",
                        prime_types)

    types, _ = _fire(minicroft, session, utterance)
    TALLY.assert_routed(utterance, f"{SKILL_ID}:{intent_label}", types)


@pytest.mark.timeout(60)
@pytest.mark.parametrize("utterance,intent_label", _GATED_CASES, ids=[c[1] for c in _GATED_CASES])
def test_gated_intent_does_not_match_fresh_session(minicroft, utterance, intent_label):
    """A brand-new session that never started a slideshow must NOT match the
    migrated intent -- this is the direction that was a permanent
    false-positive before the boot-time context leak was fixed."""
    session = _session(f"gate-negative-{intent_label}")
    types, _ = _fire(minicroft, session, utterance)
    TALLY.assert_not_routed(utterance, [f"{SKILL_ID}:{intent_label}"], types)
