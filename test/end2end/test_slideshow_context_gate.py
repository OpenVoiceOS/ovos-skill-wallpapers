"""Two-turn coverage for the "SlideShow" adapt context gate on
``NextPictureIntent``/``PrevPictureIntent``/``MakeWallpaperIntent``.

Background: those three adapt intents ``.require("SlideShow")`` (see
``__init__.py``). ``set_context("SlideShow")`` is only called from the
user-facing handlers that actually start a slideshow / show a picture
(``handle_random_wallpaper``, ``handle_random_picture``,
``handle_wallpaper_about``, ``handle_picture_about``); ``fetch_wallpapers``
itself never touches the context, so the gate opens only after the user has
done one of those things in the current session.

This module asserts both directions of that behavior:

* POSITIVE: prime a session with "show me a picture" (opens the gate), then
  fire "after" in the SAME session -- should match ``NextPictureIntent``.
* NEGATIVE: fire "after" in a brand-new, unprimed session -- should NOT
  match ``NextPictureIntent``.

The context mutates server-side (on the ``SessionManager``-registry
singleton for that ``session_id``) during turn handling, not on the local
``Session`` object passed into ``_fire()``. The POSITIVE case therefore
requires the caller's second turn to serialize the *live*, server-mutated
session rather than a stale local snapshot -- otherwise the server-side
session fold (full replace, last-writer-wins) would clobber the
context-bearing registry session with an empty one and the gate would
never open. ``_fire()`` looks the live session back up in
``SessionManager.sessions`` after each turn completes so the next turn
observes what the previous turn wrote.
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
    # The session mutates (context set) server-side during handling, on the
    # SessionManager-registry singleton for this session_id -- not on the
    # local `session` object, which stays a frozen pre-turn snapshot. Look
    # the live session back up in the registry so the caller's next turn
    # observes what this turn wrote (e.g. the SlideShow context) instead of
    # re-serializing the stale local copy.
    live_session = SessionManager.sessions.get(session.session_id, session)
    return types, live_session


@pytest.mark.timeout(90)
def test_next_picture_matches_after_priming_slideshow(minicroft):
    """Starting a slideshow in a session should open the gate for
    NextPictureIntent within that SAME session."""
    session = _session("gate-positive")
    prime_types, session = _fire(minicroft, session, "show me a picture")
    assert any(
        t.startswith(f"{SKILL_ID}:picture_random")
        for t in prime_types
    ), f"priming utterance did not route to picture_random.intent: {prime_types!r}"

    types, _ = _fire(minicroft, session, "after")
    assert any(_matches_intent(t, SKILL_ID, "NextPictureIntent") for t in types), (
        f"'after' after priming a slideshow: expected "
        f"{SKILL_ID}:NextPictureIntent, got {types!r}"
    )


@pytest.mark.timeout(60)
def test_next_picture_does_not_match_fresh_session(minicroft):
    """A brand-new session that never started a slideshow must NOT match
    NextPictureIntent -- this is the direction that was a permanent
    false-positive before the boot-time context leak was fixed."""
    session = _session("gate-negative")
    types, _ = _fire(minicroft, session, "after")
    assert not any(_matches_intent(t, SKILL_ID, "NextPictureIntent") for t in types), (
        f"'after' in a fresh, unprimed session incorrectly matched "
        f"NextPictureIntent: {types!r}"
    )
