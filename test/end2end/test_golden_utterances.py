"""Golden-utterance end-to-end coverage for ovos-skill-wallpapers (en-US).

The golden corpus (``golden_utterances_en-US.jsonl``) is a vendored slice of the
shared ovoscope golden-utterance dataset, keyed by
``skill_id == "ovos-skill-wallpapers.openvoiceos"``. One shared ``MiniCroft``
(module-scoped fixture) is booted for the whole suite; every row is its own
parametrized test item.

``next_picture.intent``, ``previous_picture.intent`` and
``make_wallpaper.intent`` are gated with ``requires_context=["SlideShow"]``
(see ``__init__.py``). The context is session-scoped and only the handlers
that start a slideshow or show a picture set it. A row for a gated intent
first primes its session with the first ``picture_random.intent`` row, then
fires the row in that same session. The wallpaper backend is stubbed, so
the priming handler never reaches the network. ``test_slideshow_context_gate.py``
asserts the gate in both directions.

Pipeline order note: this suite pins the real ovos-core default pipeline
order (padatious/padacioso-high before adapt-high, confirmed via
``Configuration()["intents"]["pipeline"]``). An adapt-high-before-padatious
ordering produces a false collision on "change the wall paper" (claimed by
``make_wallpaper.intent`` via loose ``set``/``wallpapers`` keyword overlap
instead of the intended ``wallpaper.random.intent``); pinning the real
default order avoids that test-construction artifact.
"""
import json
from pathlib import Path
from unittest.mock import patch

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

GOLDEN_PATH = Path(__file__).parent / "golden_utterances_en-US.jsonl"

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
    (adapt intents like ``next_picture.intent`` carry no ``.intent`` suffix to
    begin with; padatious/padacioso ones do)."""
    prefix = f"{skill_id}:"
    if not msg_type.startswith(prefix):
        return False
    observed = msg_type[len(prefix):]
    observed_base = observed.rsplit(".", 1)[0] if observed.endswith(".intent") else observed
    expected_base = intent_label.rsplit(".", 1)[0] if intent_label.endswith(".intent") else intent_label
    return observed_base == expected_base


# Rows that do not route, with the measured reason. Each xfail is strict:
# a row that starts passing fails the build.
#
# "after", "before" and "wall paper change" do not expand any line of
# next_picture.intent, previous_picture.intent or make_wallpaper.intent.
# The runner primes the SlideShow context before every gated row, and these
# three still come back ``ovos.intent.unmatched``.
_NOT_A_TEMPLATE = ("does not expand any line of the en-US template; unmatched "
                   "even after the SlideShow context is primed")
_XFAIL_REASONS = {
    "after": _NOT_A_TEMPLATE,
    "before": _NOT_A_TEMPLATE,
    "wall paper change": _NOT_A_TEMPLATE,
}


def _load_golden_rows():
    rows = []
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _as_param(row):
    reason = _XFAIL_REASONS.get(row["utterance"])
    if reason is None:
        return pytest.param(row, id=row["utterance"])
    return pytest.param(row, id=row["utterance"], marks=pytest.mark.xfail(reason=reason, strict=True))


_ROWS = _load_golden_rows()
GOLDEN_ROWS = [_as_param(r) for r in _ROWS]
GATED_INTENTS = {"next_picture.intent", "previous_picture.intent", "make_wallpaper.intent"}
PRIMING_UTTERANCE = next(r["utterance"] for r in _ROWS
                         if r["intent_label"] == "picture_random.intent")
FAKE_WALLPAPERS = ["/tmp/fake_wallpaper_0.jpg", "/tmp/fake_wallpaper_1.jpg"]


@pytest.fixture(scope="module")
def minicroft():
    with patch("ovos_skill_wallpapers.get_wallpapers", return_value=list(FAKE_WALLPAPERS)):
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
    """Fire one utterance; return the message types and the session the
    orchestrator stamped on ``ovos.utterance.handled``."""
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
    for m in messages:
        if m.msg_type == "ovos.utterance.handled" and m.context.get("session"):
            session = Session.deserialize(m.context["session"])
    return [m.msg_type for m in messages], session


def _golden_id(row):
    return row["utterance"]


@pytest.mark.timeout(60)
@pytest.mark.parametrize("row", GOLDEN_ROWS, ids=_golden_id)
def test_golden_utterance(minicroft, row):
    session = _fresh_session(f"golden-{_golden_id(row)}")
    if row["intent_label"] in GATED_INTENTS:
        prime_types, session = _fire(minicroft, session, PRIMING_UTTERANCE)
        assert any(_matches_intent(t, SKILL_ID, "picture_random.intent") for t in prime_types), (
            f"priming utterance {PRIMING_UTTERANCE!r} did not reach picture_random.intent: {prime_types!r}"
        )
    types, _ = _fire(minicroft, session, row["utterance"])
    assert any(_matches_intent(t, SKILL_ID, row["intent_label"]) for t in types), (
        f"{row['utterance']!r}: expected {SKILL_ID}:{row['intent_label']}, got {types!r}"
    )


@pytest.mark.timeout(60)
@pytest.mark.parametrize("negative", NEGATIVE_UTTERANCES, ids=lambda n: n[0])
def test_negative_confusable_not_claimed(minicroft, negative):
    text, source_skill = negative
    session = _fresh_session(f"negative-{text}")
    types, _ = _fire(minicroft, session, text)
    claimed = any(t.startswith(f"{SKILL_ID}:") for t in types)
    assert not claimed, f"{text!r} (from {source_skill}) was incorrectly claimed by {SKILL_ID}"
