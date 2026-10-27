"""The handler-side blacklist check folds both sides the same way.

A ``{query}`` value can arrive folded by
``ovos_spec_tools.resources.normalize_for_match``: lowercased, diacritics
stripped, ASCII punctuation stripped, so ``celle-là`` reaches the handler
as ``cellela`` and ``això`` as ``aixo``. ``_resolve_query`` must fold the
blacklist lines the same way, or every accented or hyphenated line
(``ça``, ``celle-là``, ``différente``, ``això``, ``allò``) never matches and
the deictic word is searched as a topic. Lines that are already plain
ASCII (``cela``, ``it``) match either way, which is why en-US never
showed it.

One MiniCroft per locale, booted and stopped in turn (never two alive:
ovoscope restores the process default lang at stop, and two alive restore
out of order). Each case fires the locale's own ``wallpaper_about``
phrasing with a blacklisted deictic bound to ``{query}`` and asserts the
skill did NOT search for it. The control fires a real topic in the same
locale and asserts it IS searched.
"""
from pathlib import Path
from unittest.mock import patch

import pytest
from ovos_bus_client.message import Message
from ovos_bus_client.session import Session
from ovoscope import CaptureSession
from ovoscope.routing import PIPELINE, TALLY, get_prototype_minicroft

SKILL_ID = "skill-ovos-wallpapers.openvoiceos"
LOCALE = Path(__file__).resolve().parents[2] / "locale"
FAKE = ["/tmp/fake_wallpaper_0.jpg"]
ABOUT_INTENTS = [f"{SKILL_ID}:wallpaper_about", f"{SKILL_ID}:picture_about"]
#: m2v drops an about label when the utterance holds a ``query.blacklist``
#: line (OVOS-INTENT-2 §4.3), so a deictic request may reach the random
#: handler instead; that handler searches no topic either
DEICTIC_INTENTS = ABOUT_INTENTS + [f"{SKILL_ID}:wallpaper_random"]

#: (lang, utterance with a blacklisted deictic in the {query} position,
#:  the blacklist line it binds, a real-topic utterance for the control).
#: Every deictic here carries an accent or a hyphen, the two things
#: normalize_for_match removes.
CASES = [
    ("sv-SE", "byt min bakgrundsbild om den här", "den här", "byt min bakgrundsbild om berg"),
    ("sv-SE", "byt min bakgrundsbild om den där", "den där", "byt min bakgrundsbild om berg"),
    ("fr-FR", "change le fond d'écran en ça", "ça", "change le fond d'écran en montagnes"),
    ("fr-FR", "change le fond d'écran en celle-là", "celle-là", "change le fond d'écran en montagnes"),
    ("fr-FR", "change le fond d'écran en différente", "différente", "change le fond d'écran en montagnes"),
    ("ca-ES", "canvia el fons de pantalla per això", "això", "canvia el fons de pantalla per muntanyes"),
    ("ca-ES", "canvia el fons de pantalla per allò", "allò", "canvia el fons de pantalla per muntanyes"),
]


def _fire(mc, mocked, text, lang):
    mocked.reset_mock()
    session = Session(f"fold-{lang}-{text}")
    session.lang = lang
    session.pipeline = list(PIPELINE)
    session.blacklisted_intents = []
    msg = Message("recognizer_loop:utterance", {"utterances": [text], "lang": lang},
                  {"session": session.serialize(), "source": "A", "destination": "B"})
    capture = CaptureSession(mc, eof_msgs=["ovos.utterance.handled", "ovos.intent.unmatched"])
    capture.capture(msg, timeout=30)
    types = [m.msg_type for m in capture.finish()]
    query = None
    if mocked.call_args is not None:
        args, kwargs = mocked.call_args
        query = kwargs.get("query", args[0] if args else None)
    return types, query


@pytest.mark.timeout(240)
@pytest.mark.parametrize("lang, deictic_utterance, line, topic_utterance", CASES,
                         ids=[f"{c[0]}-{c[2]}" for c in CASES])
def test_an_accented_or_hyphenated_blacklist_line_still_blocks_the_query(
        lang, deictic_utterance, line, topic_utterance):
    blacklist = LOCALE / lang / "query.blacklist"
    assert line in blacklist.read_text(encoding="utf-8").splitlines(), (
        f"{line!r} is not a line of {lang}/query.blacklist; the case is stale")
    with patch("ovos_skill_wallpapers.get_wallpapers", return_value=list(FAKE)) as mocked:
        mc = get_prototype_minicroft([SKILL_ID], lang=lang, max_wait=150)
        try:
            types, query = _fire(mc, mocked, deictic_utterance, lang)
            TALLY.assert_routed(f"[{lang}] {deictic_utterance}", DEICTIC_INTENTS, types)
            assert query is None or not _folds_to(query, line), (
                f"[{lang}] the blacklisted {line!r} was searched as a topic, as {query!r}")
            # the control: a real topic in the same locale is searched
            types, topic = _fire(mc, mocked, topic_utterance, lang)
            TALLY.assert_routed(f"[{lang}] {topic_utterance}", ABOUT_INTENTS, types)
            assert topic, f"[{lang}] the control {topic_utterance!r} searched nothing"
        finally:
            mc.stop()


def _folds_to(query, line):
    from ovos_spec_tools.resources import normalize_for_match
    return normalize_for_match(query) == normalize_for_match(line)
