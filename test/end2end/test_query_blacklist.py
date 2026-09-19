"""Real bus round-trip coverage for the ``{query}`` slot blacklist
(``locale/en-US/query.blacklist``) and its handler-side fallback.

``ovos_workshop.skills.ovos.OVOSSkill.register_intent_file`` (OVOS-INTENT-2
§4.3) reads every ``{slot}`` in a padatious ``.intent`` template and, for
each one, loads a sibling ``<slot>.blacklist`` file of values that must not
bind to that slot. ``wallpaper_about.intent`` and ``picture_about.intent``
both declare ``{query}``, and ``query.blacklist`` lists anaphoric fillers
("that", "this", "it", ...) so a bare deictic reference does not get treated
as a search topic.

Padatious does not refuse to match the intent when a bound slot value is
blacklisted -- it unresolves the slot (OVOS-INTENT-2 §4.3: "treating as
unresolved") and the intent still fires with ``query`` missing from
``match_data``. A confident (high-tier) match never reaches this path at
all for these short deictic utterances -- padatious's own classifier
confidence for a bare single-word slot at the end of the template does not
clear the high threshold with this skill's small sample set, so the intent
simply does not match there, blacklist or not. The blacklist only makes an
observable difference once the utterance clears the *medium* confidence
tier, which is why the padatious arm below pins ``high`` + ``medium``
rather than just ``high`` -- pinning only ``high`` was a false-green:
deleting ``query.blacklist`` entirely did not flip any assertion.

``<slot>.blacklist`` is a padatious-specific mechanism -- ``register_template``
only forwards ``slot_blacklist`` over the legacy ``padatious:register_intent``
bus topic, never over the spec-compliant ``ovos.intent.register.template``
message that padacioso/adapt consume (see
``IntentServiceInterface.register_template`` in ovos-workshop). padacioso has
no slot-blacklist support at all, so with the padacioso pipeline the deictic
filler DOES bind to ``{query}`` at match time; the only thing standing
between that raw binding and a bogus search is ``_resolve_query`` in
``__init__.py``, which re-checks the bound value against
``query.blacklist`` itself and falls back to the random-wallpaper/picture
path when it is blacklisted or the slot came back empty. The padatious arm
below exercises the blacklist at match time; the padacioso arm exercises
``_resolve_query`` as the sole line of defense -- both are needed, since a
suite that only runs padatious gives ``_resolve_query`` zero coverage.

This suite fires real utterances through a ``MiniCroft`` and asserts on
what the skill actually searched for (via the ``get_wallpapers`` call the
handler makes), not on the bus message types alone.
"""
from unittest.mock import patch

import pytest
from ovos_bus_client.message import Message
from ovos_bus_client.session import Session
from ovoscope import CaptureSession, get_minicroft

SKILL_ID = "skill-ovos-wallpapers.openvoiceos"
LANG = "en-US"

FAKE_WALLPAPERS = ["/tmp/fake_wallpaper_0.jpg", "/tmp/fake_wallpaper_1.jpg"]

# Two pipeline arms, both parametrized through every test below:
#
# - padatious: the blacklist is enforced by padatious itself at match time
#   (OVOS-INTENT-2 §4.3 per-slot ``<slot>.blacklist``); high is included
#   alongside medium because a high-tier match never reaches this path for
#   these short deictic utterances (see module docstring), so pinning only
#   high would hide the blacklist entirely.
# - padacioso: padacioso has no slot-blacklist support, so the deictic word
#   binds to {query} unfiltered at match time; ``_resolve_query`` in the
#   skill is the only thing that still catches it.
_PIPELINES = {
    "padatious": [
        "ovos-padatious-pipeline-plugin-high",
        "ovos-padatious-pipeline-plugin-medium",
    ],
    "padacioso": [
        "ovos-padacioso-pipeline-plugin-high",
        "ovos-padacioso-pipeline-plugin-medium",
    ],
}


@pytest.fixture(scope="module")
def minicroft():
    with patch("ovos_skill_wallpapers.get_wallpapers", return_value=list(FAKE_WALLPAPERS)) as mocked:
        mc = get_minicroft([SKILL_ID])
        yield mc, mocked
        mc.stop()


def _fire(mc, mocked, text, pipeline):
    mocked.reset_mock()
    session = Session(f"blacklist-{text}-{pipeline[0]}")
    session.lang = LANG
    session.pipeline = list(pipeline)
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
@pytest.mark.parametrize("pipeline_name", list(_PIPELINES))
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
def test_deictic_query_is_not_claimed_as_topic(minicroft, text, pipeline_name):
    """A bare deictic reference must never be searched for as if it were a
    real topic, regardless of which pipeline stage claims the intent. With
    padatious the blacklist unresolves the slot at match time; with
    padacioso the slot binds raw and ``_resolve_query`` must catch it in the
    handler instead. Either way the intent can still fire -- the handler
    must never crash and must never hand the deictic word itself to
    ``get_wallpapers``."""
    mc, mocked = minicroft
    types, query = _fire(mc, mocked, text, _PIPELINES[pipeline_name])
    assert "mycroft.skill.handler.error" not in types, (
        f"{text!r} ({pipeline_name}) crashed the handler: {types!r}"
    )
    assert query is None, (
        f"{text!r} ({pipeline_name}) was searched for as a literal topic: "
        f"get_wallpapers(query={query!r})"
    )


@pytest.mark.timeout(60)
@pytest.mark.parametrize("pipeline_name", list(_PIPELINES))
def test_real_topic_query_is_claimed(minicroft, pipeline_name):
    """Sanity check: a real topic in the same slot position still matches
    and is actually searched for, so the blacklist/fallback path is
    filtering the deictic fillers specifically, not breaking the intent."""
    mc, mocked = minicroft
    types, query = _fire(mc, mocked, "set the wallpaper to space", _PIPELINES[pipeline_name])
    assert any(t.startswith(f"{SKILL_ID}:wallpaper_about") for t in types), (
        f"real topic query was not routed to wallpaper_about ({pipeline_name}): {types!r}"
    )
    assert query == "space", (
        f"expected get_wallpapers(query='space') ({pipeline_name}), got {query!r}"
    )


@pytest.mark.timeout(60)
@pytest.mark.parametrize("pipeline_name", list(_PIPELINES))
def test_deictic_phrase_with_real_topic_is_claimed(minicroft, pipeline_name):
    """Both the padatious blacklist and the ``_resolve_query`` fallback
    match by whole-value equality, not substring -- "that beach" is not
    "that", so a deictic word merely appearing inside a longer, genuine
    topic must still bind and be searched for verbatim."""
    mc, mocked = minicroft
    types, query = _fire(mc, mocked, "set the wallpaper to that beach", _PIPELINES[pipeline_name])
    assert any(t.startswith(f"{SKILL_ID}:wallpaper_about") for t in types), (
        f"'that beach' query was not routed to wallpaper_about ({pipeline_name}): {types!r}"
    )
    assert query == "that beach", (
        f"expected get_wallpapers(query='that beach') ({pipeline_name}), got {query!r}"
    )
