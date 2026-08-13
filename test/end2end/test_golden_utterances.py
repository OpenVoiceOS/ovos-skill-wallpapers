"""Golden-utterance end-to-end coverage for ovos-skill-wallpapers (en-US).

The golden corpus (``golden_utterances.jsonl``) is a vendored slice of the
shared ovoscope golden-utterance dataset, keyed by
``skill_id == "ovos-skill-wallpapers.openvoiceos"``. One shared ``MiniCroft``
(module-scoped fixture) is booted for the whole suite; every row is its own
parametrized test item.

Runtime note: the corpus keys rows by the PyPI/repo-derived skill_id
(``ovos-skill-wallpapers.openvoiceos``), but the skill's actual OPM entry
point -- and therefore the routed message prefix -- is
``skill-ovos-wallpapers.openvoiceos`` (see ``setup.py``'s
``PLUGIN_ENTRY_POINT`` and ``test_intents_en_us.py``). This suite asserts
against the runtime id, not the corpus label.

``NextPictureIntent``, ``PrevPictureIntent`` and ``MakeWallpaperIntent`` are
adapt intents that ``.require("SlideShow")`` (see ``__init__.py``). This is
now genuinely session-scoped: ``self.set_context("SlideShow")`` is only set
by the user-facing handlers that actually start a slideshow or show a
picture/wallpaper (``handle_random_wallpaper``, ``handle_random_picture``,
``handle_wallpaper_about``, ``handle_picture_about``). It used to also be
set as a side effect of ``initialize()`` self-emitting
``"{skill_id}.get.wallpaper.collection"`` on skill boot (wired to
``handle_wallpaper_scan`` -> ``fetch_wallpapers``, which used to call
``set_context`` unconditionally on every fetch, including the boot-time
PHAL collection scan) -- so "SlideShow" used to be set globally the moment
the skill loaded, before any utterance was ever fired, making the context
guard a permanent no-op. That leak has been fixed: ``fetch_wallpapers`` no
longer sets the context itself, so a brand-new session with no prior
slideshow-starting utterance correctly does NOT match these intents (see
the now-xfailed "after"/"before"/"wall paper change" rows below, and
``test_slideshow_context_gate.py`` for dedicated two-turn coverage of the
fixed behavior). Real two-turn (prime-then-gate) coverage cannot go fully
green yet because context-gated matching for TEXT queries is currently
broken upstream (core#857 pending); see that test module's own xfail
markers and reasons for exactly which direction is blocked and which
already passes.

Pipeline order note: an adapt-high-before-padatious/padacioso-high test
pipeline was tried first and produced a false collision on "change the wall
paper" (claimed by ``MakeWallpaperIntent`` via loose ``set``/``wallpapers``
keyword overlap instead of the intended ``wallpaper.random.intent``). That
collision does not reproduce under the real ovos-core default pipeline
order (padatious/padacioso-high before adapt-high, confirmed via
``Configuration()["intents"]["pipeline"]``), which is what this suite uses
-- so it was a test-construction artifact, not a skill defect.
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
# high BEFORE adapt-high -- see Configuration()["intents"]["pipeline"]), not
# an arbitrary order: an adapt-high-first ordering was tried first and
# produced a false collision (see PR description) that does not reproduce
# under the real default order.
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
# -turn golden rows fail as written. Priming with a "show me a picture"
# utterance first would normally fix this, but that two-turn path is
# currently blocked upstream by core#857 (context-gated matching is broken
# for text queries) -- see ``test_slideshow_context_gate.py`` for the
# two-turn coverage and its own xfail markers.
_XFAIL_REASONS = {
    "after": "requires SlideShow context from a prior priming utterance; "
             "single-turn golden row has no priming and two-turn priming is "
             "blocked by core#857 (see test_slideshow_context_gate.py)",
    "before": "requires SlideShow context from a prior priming utterance; "
              "single-turn golden row has no priming and two-turn priming is "
              "blocked by core#857 (see test_slideshow_context_gate.py)",
    "wall paper change": "requires SlideShow context from a prior priming "
                          "utterance; single-turn golden row has no priming "
                          "and two-turn priming is blocked by core#857 (see "
                          "test_slideshow_context_gate.py)",
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
