"""Multilingual golden-utterance end-to-end coverage for ovos-skill-wallpapers.

Every locale that ships real intent/vocab content gets its own
golden_utterances_<lang>.jsonl, rows expanded directly from that locale's
own .intent templates (alternations/optionals resolved) and {query} slots
filled from that locale's own query.entity values.

``next_picture.intent``, ``previous_picture.intent`` and
``make_wallpaper.intent`` require the "SlideShow" context. A row for one of
them first primes its session with the locale's own first
``picture_random.intent`` row, which opens the gate, and then fires the row
in that same session (the two-turn shape of test_slideshow_context_gate.py).
The wallpaper backend is stubbed, so the priming handler never reaches the
network.

The locales come from the golden files on disk, and collection fails unless
the set of golden files equals the set of locale directories that ship an
``.intent`` file.

One MiniCroft is booted per locale in turn (lang=<locale>, no
secondary_langs -- see ovos-skill-date-time/test/end2end/test_intents_it_it.py
on dev).
"""
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from ovos_bus_client.message import Message
from ovos_bus_client.session import Session
from ovoscope import CaptureSession, get_minicroft

SKILL_ID = "skill-ovos-wallpapers.openvoiceos"

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

END2END_DIR = Path(__file__).parent
LOCALE_DIR = END2END_DIR.parent.parent / "locale"

GOLDEN_LANGS = {p.stem.split("golden_utterances_", 1)[1]
                for p in END2END_DIR.glob("golden_utterances_*.jsonl")}
INTENT_LANGS = {d.name for d in LOCALE_DIR.iterdir()
                if d.is_dir() and any(d.glob("*.intent"))}
assert GOLDEN_LANGS == INTENT_LANGS, (
    f"golden files without an .intent locale: {sorted(GOLDEN_LANGS - INTENT_LANGS)}; "
    f".intent locales without a golden file: {sorted(INTENT_LANGS - GOLDEN_LANGS)}"
)

# en-US runs in test_golden_utterances.py.
EXCLUDED_LANGS = {"en-US"}
LANGS = sorted(GOLDEN_LANGS - EXCLUDED_LANGS)
assert LANGS, "no golden_utterances_<lang>.jsonl files found"

GATED_INTENTS = {"next_picture.intent", "previous_picture.intent", "make_wallpaper.intent"}


def _matches_intent(msg_type: str, skill_id: str, intent_label: str) -> bool:
    prefix = f"{skill_id}:"
    if not msg_type.startswith(prefix):
        return False
    observed = msg_type[len(prefix):]
    observed_base = observed.rsplit(".", 1)[0] if observed.endswith(".intent") else observed
    expected_base = intent_label.rsplit(".", 1)[0] if intent_label.endswith(".intent") else intent_label
    return observed_base == expected_base


def _load_rows(lang):
    path = END2END_DIR / f"golden_utterances_{lang}.jsonl"
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert rows, f"{lang}: no golden rows"
    return rows


ROWS_BY_LANG = {lang: _load_rows(lang) for lang in LANGS}
ALL_ROWS = [row for lang in LANGS for row in ROWS_BY_LANG[lang]]
PRIMING_UTTERANCE = {}
for _lang, _rows in ROWS_BY_LANG.items():
    _priming = [r["utterance"] for r in _rows if r["intent_label"] == "picture_random.intent"]
    assert _priming, f"{_lang}: no picture_random.intent row to prime the SlideShow context"
    PRIMING_UTTERANCE[_lang] = _priming[0]

FAKE_WALLPAPERS = ["/tmp/fake_wallpaper_0.jpg", "/tmp/fake_wallpaper_1.jpg"]


def _as_param(row):
    gated = "gated" if row["intent_label"] in GATED_INTENTS else "tier1"
    return pytest.param(row, id=f"{row['lang']}-{gated}-{row['intent_label']}-{row['utterance']}")


GOLDEN_ROWS = [_as_param(r) for r in ALL_ROWS]

_BOOTED = {}


@pytest.fixture(scope="module")
def mc_factory(request):
    # picture_about.intent/wallpaper_about.intent/picture_random.intent have
    # many nested alternation groups; some locales (ca-ES, es-ES) need more
    # than the ovoscope trained-wait default to finish padatious training
    # under CI resource constraints -- same heavy-skill convention as
    # ovos-skill-alerts' conftest.py.
    os.environ.setdefault("OVOSCOPE_TRAINED_TIMEOUT", "300")
    backend = patch("ovos_skill_wallpapers.get_wallpapers", return_value=list(FAKE_WALLPAPERS))
    backend.start()
    request.addfinalizer(backend.stop)

    def _get(lang):
        if lang not in _BOOTED:
            mc = get_minicroft([SKILL_ID], max_wait=300, lang=lang)
            _BOOTED[lang] = mc
            request.addfinalizer(mc.stop)
        return _BOOTED[lang]
    return _get


def _session(session_id, lang):
    session = Session(session_id)
    session.lang = lang
    session.pipeline = list(_PIPELINE)
    session.blacklisted_intents = []
    return session


def _fire(mc, session, text, lang):
    """Fire one utterance; return the message types and the session the
    orchestrator stamped on ``ovos.utterance.handled``."""
    utterance = Message(
        "recognizer_loop:utterance",
        {"utterances": [text], "lang": lang},
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
    return f"{row['lang']}-{row['intent_label']}-{row['utterance']}"


# These rows do not expand their locale's template. The templates join two
# alternation groups with no space between them, for example
# ``(pantalla de inicio|fondo de pantalla)(aleatorio|nuevo)`` in es-ES
# wallpaper_random.intent and ``(nueva|aleatoria|)(sobre|para|con)`` in es-ES
# picture_about.intent, so the trained sentence fuses two words
# ("inicioaleatorio", "nuevasobre"). ca-ES has the same join in
# wallpaper_random.intent ``(un|una|el|la|es|sa|)(nou|nova|)`` and in
# picture_random.intent ``mostra('m|)`` and ``(nou|nova|aleatori|aleatòria)
# (imatge|...)``. The rows write the words apart, and padatious does not
# match them. The fix belongs in the locale files.
KNOWN_BUGS = {
    ("ca-ES", "canvia un nou fons nou aleatori"): "ca-ES wallpaper_random.intent joins two groups with no space; the row writes the words apart",
    ("ca-ES", "posa una nova fons de pantalla nova aleatòria"): "ca-ES wallpaper_random.intent joins two groups with no space; the row writes the words apart",
    ("ca-ES", "mostra 'm un altre nou imatge"): "ca-ES picture_random.intent joins two groups with no space; the row writes the words apart",
    ("es-ES", "pantalla de inicio aleatorio"): "es-ES wallpaper_random.intent joins two groups with no space; the row writes the words apart",
    ("es-ES", "muestra imagen nueva sobre naturaleza"): "es-ES picture_about.intent joins two groups with no space; the row writes the words apart",
    ("es-ES", "muestra imagen nueva sobre espacio"): "es-ES picture_about.intent joins two groups with no space; the row writes the words apart",
    ("es-ES", "mostrar foto aleatoria para naturaleza"): "es-ES picture_about.intent joins two groups with no space; the row writes the words apart",
}


@pytest.mark.timeout(400)
@pytest.mark.parametrize("row", GOLDEN_ROWS, ids=_golden_id)
def test_golden_utterance_multilang(mc_factory, row):
    lang = row["lang"]
    mc = mc_factory(lang)
    session = _session(f"golden-{_golden_id(row)}", lang)
    if row["intent_label"] in GATED_INTENTS:
        prime_types, session = _fire(mc, session, PRIMING_UTTERANCE[lang], lang)
        assert any(_matches_intent(t, SKILL_ID, "picture_random.intent") for t in prime_types), (
            f"[{lang}] priming utterance {PRIMING_UTTERANCE[lang]!r} did not reach "
            f"picture_random.intent: {prime_types!r}"
        )
    types, _ = _fire(mc, session, row["utterance"], lang)
    matched = any(_matches_intent(t, SKILL_ID, row["intent_label"]) for t in types)
    bug_key = (row["lang"], row["utterance"])
    if bug_key in KNOWN_BUGS and not matched:
        pytest.xfail(reason=f"known-bug: {KNOWN_BUGS[bug_key]}")
    assert matched, (
        f"[{row['lang']}] {row['utterance']!r}: expected {SKILL_ID}:{row['intent_label']}, got {types!r}"
    )
