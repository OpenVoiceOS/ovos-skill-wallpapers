"""Golden-utterance end-to-end coverage for ovos-skill-wallpapers (en-US).

The golden corpus (``golden_utterances.jsonl``) is a vendored slice of the
shared ovoscope golden-utterance dataset, keyed by
``skill_id == "ovos-skill-wallpapers.openvoiceos"``. One shared ``MiniCroft``
(module-scoped fixture) is booted for the whole suite; every row is its own
parametrized test item.

``NextPictureIntent``, ``PrevPictureIntent`` and ``MakeWallpaperIntent`` are
adapt intents that ``.require("SlideShow")`` (see ``__init__.py``). The
context is session-scoped: ``self.set_context("SlideShow")`` is only set by
the user-facing handlers that actually start a slideshow or show a
picture/wallpaper (``handle_random_wallpaper``, ``handle_random_picture``,
``handle_wallpaper_about``, ``handle_picture_about``); ``fetch_wallpapers``
itself never sets the context, so a boot-time collection scan does not leak
"SlideShow" globally. A brand-new session with no prior slideshow-starting
utterance correctly does NOT match these intents, which is why the
"after"/"before"/"wall paper change" rows below are xfailed here: they fire
the utterance with no priming turn at all. ``test_slideshow_context_gate.py``
covers the two-turn priming path (prime the context, then fire the follow-up)
that these single-turn golden rows do not exercise.

Pipeline order note: this suite pins the real ovos-core default pipeline
order (padatious/padacioso-high before adapt-high, confirmed via
``Configuration()["intents"]["pipeline"]``). An adapt-high-before-padatious
ordering produces a false collision on "change the wall paper" (claimed by
``MakeWallpaperIntent`` via loose ``set``/``wallpapers`` keyword overlap
instead of the intended ``wallpaper.random.intent``); pinning the real
default order avoids that test-construction artifact.
"""
import json
from pathlib import Path

import pytest
from ovos_bus_client.message import Message
from ovos_bus_client.session import Session
from ovoscope import CaptureSession, get_minicroft

SKILL_ID = "skill-ovos-wallpapers.openvoiceos"
LANG = "en-US"

# Matches the real default ovos-core pipeline order (padatious/padacioso
# high BEFORE adapt-high -- see Configuration()["intents"]["pipeline"]).
# An adapt-high-first ordering produces a false collision on "change the
# wall paper" that does not reproduce under this, the real default order.
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

GOLDEN_PATH = Path(__file__).parent / "golden_utterances.jsonl"

# utterances lifted verbatim from OTHER skills' golden-utterance slices in
# the shared ovoscope corpus, picked for lexical overlap with wallpapers'
# "show me"/"change"/"picture"/"display" vocabulary.
NEGATIVE_UTTERANCES = [
    ("can you tell me the weather", "ovos-skill-weather.openvoiceos"),
    ("can you find something on wikipedia", "ovos-skill-wikipedia.openvoiceos"),
    ("tell me the word of the day", "ovos-skill-word-of-the-day.openvoiceos"),
    ("search wikihow for something", "ovos-skill-wikihow.openvoiceos"),
    ("ask wordnet about word", "ovos-skill-wordnet.openvoiceos"),
    ("set an alarm", "ovos-skill-alerts.openvoiceos"),
    ("can you spell word", "ovos-skill-spelling.openvoiceos"),
    # lexical near-miss on this skill's own "set [my|the] (wallpaper|wall
    # paper) to {query}" template ("background" vs "wallpaper"/"wall
    # paper"); not sourced from another skill's corpus.
    ("set the background to blue", "ovos-skill-wallpapers.openvoiceos"),
]


def _matches_intent(msg_type: str, skill_id: str, intent_label: str) -> bool:
    """Tolerant matcher, same shape as the sibling repos' suites: compare
    the ``:``-suffix basename, extension-stripped and case/punct-insensitive
    (adapt intents like ``NextPictureIntent`` carry no ``.intent`` suffix to
    begin with; padatious/padacioso ones do)."""
    prefix = f"{skill_id}:"
    if not msg_type.startswith(prefix):
        return False
    observed = msg_type[len(prefix):]
    observed_base = observed.rsplit(".", 1)[0] if observed.endswith(".intent") else observed
    expected_base = intent_label.rsplit(".", 1)[0] if intent_label.endswith(".intent") else intent_label
    return observed_base == expected_base


# Rows that do not currently route correctly, with the root-caused reason.
# All xfails are strict=True: a row that starts passing must fail the build.
#
# "after"/"before"/"wall paper change" require the "SlideShow" adapt context
# (see NextPictureIntent/PrevPictureIntent/MakeWallpaperIntent in
# ``__init__.py``). Until the boot-time context leak was fixed, the context
# was set globally on skill load and these single-turn rows (no priming
# utterance) matched unconditionally -- a false green documented in the old
# module docstring. Now that the context is only set by a user-facing
# handler that actually starts a slideshow/shows a picture, a fresh,
# unprimed session correctly does NOT match these intents, so these single
# -turn golden rows fail as written -- they are single-turn by construction
# (the golden corpus fires one utterance per row) and were never expected to
# carry priming context. That is unrelated to two-turn priming, which is
# NOT blocked upstream (see test_slideshow_context_gate.py, which covers
# and passes the two-turn prime-then-gate path with no core/workshop
# changes needed).
_XFAIL_REASONS = {
    "after": "requires SlideShow context from a prior priming utterance; "
             "this golden row is single-turn by construction and carries no "
             "priming turn (see test_slideshow_context_gate.py for two-turn "
             "coverage of the primed case, which passes)",
    "before": "requires SlideShow context from a prior priming utterance; "
              "this golden row is single-turn by construction and carries no "
              "priming turn (see test_slideshow_context_gate.py for two-turn "
              "coverage of the primed case, which passes)",
    "wall paper change": "requires SlideShow context from a prior priming "
                          "utterance; this golden row is single-turn by "
                          "construction and carries no priming turn (see "
                          "test_slideshow_context_gate.py for two-turn "
                          "coverage of the primed case, which passes)",
}


def _load_golden_rows():
    rows = []
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("needs_manual"):
                continue
            rows.append(row)
    return rows


def _as_param(row):
    reason = _XFAIL_REASONS.get(row["utterance"])
    if reason is None:
        return pytest.param(row, id=row["utterance"])
    return pytest.param(row, id=row["utterance"], marks=pytest.mark.xfail(reason=reason, strict=True))


GOLDEN_ROWS = [_as_param(r) for r in _load_golden_rows()]


@pytest.fixture(scope="module")
def minicroft():
    mc = get_minicroft([SKILL_ID])
    yield mc
    mc.stop()


def _fresh_session(session_id):
    session = Session(session_id)
    session.lang = LANG
    session.pipeline = list(_PIPELINE)
    # blacklisted_intents defaults to None on a fresh Session, which crashes
    # the padacioso pipeline (NoneType membership test) - force an empty list.
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
    return [m.msg_type for m in capture.finish()]


def _golden_id(row):
    return row["utterance"]


@pytest.mark.timeout(60)
@pytest.mark.parametrize("row", GOLDEN_ROWS, ids=_golden_id)
def test_golden_utterance(minicroft, row):
    session = _fresh_session(f"golden-{_golden_id(row)}")
    types = _fire(minicroft, session, row["utterance"])
    assert any(_matches_intent(t, SKILL_ID, row["intent_label"]) for t in types), (
        f"{row['utterance']!r}: expected {SKILL_ID}:{row['intent_label']}, got {types!r}"
    )


@pytest.mark.timeout(60)
@pytest.mark.parametrize("negative", NEGATIVE_UTTERANCES, ids=lambda n: n[0])
def test_negative_confusable_not_claimed(minicroft, negative):
    text, source_skill = negative
    session = _fresh_session(f"negative-{text}")
    types = _fire(minicroft, session, text)
    claimed = any(t.startswith(f"{SKILL_ID}:") for t in types)
    assert not claimed, f"{text!r} (from {source_skill}) was incorrectly claimed by {SKILL_ID}"
