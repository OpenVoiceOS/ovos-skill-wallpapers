"""Two-turn coverage for the "SlideShow" adapt context gate on
``NextPictureIntent``/``PrevPictureIntent``/``MakeWallpaperIntent``.

Background: those three adapt intents ``.require("SlideShow")`` (see
``__init__.py``). Until fixed, ``initialize()`` self-emitted
``"{skill_id}.get.wallpaper.collection"`` at skill boot, which chained
through ``handle_wallpaper_scan`` -> ``fetch_wallpapers`` ->
``self.set_context("SlideShow")`` -- setting the context globally before
any utterance was ever fired, making the gate a permanent no-op. The fix
moves ``set_context("SlideShow")`` out of ``fetch_wallpapers`` and into the
user-facing handlers that actually start a slideshow / show a picture
(``handle_random_wallpaper``, ``handle_random_picture``,
``handle_wallpaper_about``, ``handle_picture_about``), so the gate should
now only open after the user has done that in the current session.

This module asserts both directions of that intended behavior:

* POSITIVE: prime a session with "show me a picture" (opens the gate), then
  fire "after" in the SAME session -- should match ``NextPictureIntent``.
* NEGATIVE: fire "after" in a brand-new, unprimed session -- should NOT
  match ``NextPictureIntent`` (this is the direction that used to be a
  permanent false-positive before the fix).

Both directions pass. The POSITIVE direction was previously marked xfail,
blaming a supposed upstream text-query context-matching defect (core#857).
That diagnosis was wrong: the real bug was in this test's own ``_fire()``
helper, which returned the *local* pre-turn ``Session`` object instead of
the live, server-mutated session for that ``session_id``. Turn 1 sets the
"SlideShow" context on the ``SessionManager``-registry singleton, but the
stale local object handed back to the caller never saw that mutation. Turn
2 then serialized that stale, empty snapshot into its outgoing Message, and
the server-side session fold (full replace, last-writer-wins) clobbered the
registry's context-bearing session with it -- so "after" was evaluated
against a session that had never been primed, and the gate never opened.
``_fire()`` now looks the live session back up in ``SessionManager.sessions``
after the turn completes, and the POSITIVE assertion passes with no core or
workshop changes required.
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
    # the session mutates (context set) server-side during handling, on the
    # SessionManager-registry singleton for this session_id -- NOT on the
    # local `session` object, which stays a frozen pre-turn snapshot. Look
    # the live session back up in the registry so the caller's next turn
    # actually observes what this turn wrote (e.g. the SlideShow context),
    # instead of re-serializing the stale local copy and having the
    # server-side fold (full replace, last-writer-wins) wipe it out again.
    live_session = SessionManager.sessions.get(session.session_id, session)
    return types, live_session


@pytest.mark.timeout(90)
def test_next_picture_matches_after_priming_slideshow(minicroft):
    """Starting a slideshow in a session should open the gate for
    NextPictureIntent within that SAME session."""
    session = _session("gate-positive")
    prime_types, session = _fire(minicroft, session, "show me a picture")
    assert any(
        t.startswith(f"{SKILL_ID}:picture.random")
        for t in prime_types
    ), f"priming utterance did not route to picture.random.intent: {prime_types!r}"

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
