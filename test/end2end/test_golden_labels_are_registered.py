"""Every golden row must name an intent this skill registers.

A golden row whose ``intent_label`` names nothing is not caught by the
golden suite when that row is an expected failure: the row never routes, so
the label is never compared. Three rows carried ``MakeWallpaperIntent``,
``NextPictureIntent`` and ``PrevPictureIntent``, names that no file and no
handler in this repository ever used, and a corpus built from the same
labels trained three classes no skill answers to.

This module reads the labels against the ``.intent`` files on disk, so it
holds for an expected-failure row exactly as it does for a passing one.
"""
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GOLDEN = REPO / "test" / "end2end" / "golden_utterances_en-US.jsonl"
LOCALE = REPO / "locale" / "en-US"


def _rows():
    with GOLDEN.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _shipped_intents():
    return {p.name for p in LOCALE.glob("*.intent")}


@pytest.mark.parametrize("row", list(_rows()), ids=lambda r: r["utterance"])
def test_the_label_names_a_shipped_intent_file(row):
    label = row["intent_label"]
    assert label.endswith(".intent"), (
        f"{row['utterance']!r}: {label!r} is not a resource file name")
    assert label in _shipped_intents(), (
        f"{row['utterance']!r}: {label!r} is not shipped in locale/en-US; "
        f"the skill has {sorted(_shipped_intents())}")


def test_the_locale_is_not_empty():
    """The control: an empty glob would make every case above vacuous."""
    assert _shipped_intents()


def test_a_name_no_skill_registers_is_refused():
    """The control in the other direction, on the check itself."""
    assert "MakeWallpaperIntent" not in _shipped_intents()
    assert "MakeWallpaperIntent.intent" not in _shipped_intents()
