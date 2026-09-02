"""Two-turn coverage for the "SlideShow" context gate on
``next_picture.intent``/``previous_picture.intent``/``make_wallpaper.intent``.

Those three intents are now defined as ``.intent`` files (padatious/
padacioso), gated via ``@intent_handler(..., requires_context=["SlideShow"])``
(see ``__init__.py``) instead of the previous ``IntentBuilder(...).require(
"SlideShow")`` adapt construction. ``self.set_context("SlideShow")`` is only
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

``_fire()`` looks the live session back up in ``SessionManager.sessions``
after a turn completes instead of returning the caller's stale local
``Session`` snapshot: the context is mutated server-side on the
``SessionManager``-registry singleton for that ``session_id``, and handing
back the frozen pre-turn object would let the server-side session fold
(full replace, last-writer-wins) clobber the freshly primed context on the
next turn.
"""
import pytest
from ovos_bus_client.message import Message
from ovos_bus_client.session import Session, SessionManager
from ovoscope import CaptureSession, get_minicroft

SKILL_ID = "skill-ovos-wallpapers.openvoiceos"
LANG = "en-US"

_PIPELINE = [
    "ovos-padatious-pipeline-plugin-high",
    "ovos-padacioso-pipeline-plugin-high",
    "ovos-adapt-pipeline-plugin-high",
    "ovos-padacioso-pipeline-plugin-medium",
    "ovos-adapt-pipeline-plugin-medium",
]

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


def _matches_intent(msg_type: str, skill_id: str, intent_label: str) -> bool:
    prefix = f"{skill_id}:"
    if not msg_type.startswith(prefix):
        return False
    observed = msg_type[len(prefix):]
    observed_base = observed.rsplit(".", 1)[0] if observed.endswith(".intent") else observed
    expected_base = intent_label.rsplit(".", 1)[0] if intent_label.endswith(".intent") else intent_label
    return observed_base == expected_base


@pytest.fixture(scope="module")
def minicroft():
    mc = get_minicroft([SKILL_ID])
    yield mc
    mc.stop()


def _session(session_id):
    session = Session(session_id)
    session.lang = LANG
    session.pipeline = list(_PIPELINE)
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
        eof_msgs=["mycroft.skill.handler.start", "ovos.intent.unmatched"],
        ignore_messages=_IGNORE,
    )
    capture.capture(utterance, timeout=30)
    types = [m.msg_type for m in capture.finish()]
    live_session = SessionManager.sessions.get(session.session_id, session)
    return types, live_session


@pytest.mark.timeout(90)
@pytest.mark.parametrize("utterance,intent_label", _GATED_CASES, ids=[c[1] for c in _GATED_CASES])
def test_gated_intent_matches_after_priming_slideshow(minicroft, utterance, intent_label):
    """Starting a slideshow in a session should open the gate for the
    migrated intent within that SAME session."""
    session = _session(f"gate-positive-{intent_label}")
    prime_types, session = _fire(minicroft, session, "show me a picture")
    assert any(
        t.startswith(f"{SKILL_ID}:picture_random")
        for t in prime_types
    ), f"priming utterance did not route to picture_random.intent: {prime_types!r}"

    types, _ = _fire(minicroft, session, utterance)
    assert any(_matches_intent(t, SKILL_ID, intent_label) for t in types), (
        f"{utterance!r} after priming a slideshow: expected "
        f"{SKILL_ID}:{intent_label}, got {types!r}"
    )


@pytest.mark.timeout(60)
@pytest.mark.parametrize("utterance,intent_label", _GATED_CASES, ids=[c[1] for c in _GATED_CASES])
def test_gated_intent_does_not_match_fresh_session(minicroft, utterance, intent_label):
    """A brand-new session that never started a slideshow must NOT match the
    migrated intent -- this is the direction that was a permanent
    false-positive before the boot-time context leak was fixed."""
    session = _session(f"gate-negative-{intent_label}")
    types, _ = _fire(minicroft, session, utterance)
    assert not any(_matches_intent(t, SKILL_ID, intent_label) for t in types), (
        f"{utterance!r} in a fresh, unprimed session incorrectly matched "
        f"{intent_label}: {types!r}"
    )
