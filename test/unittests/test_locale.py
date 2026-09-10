"""Unit tests over the wallpapers skill locale resources."""
import os
import re
from glob import glob
from unittest import TestCase

LOCALE = os.path.join(os.path.dirname(__file__), "..", "..", "locale")
EN_US = os.path.join(LOCALE, "en-US")


def _read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


class TestLocaleResources(TestCase):
    def test_intent_slots_are_query(self):
        """Every intent slot must be {query} to match message.data['query']."""
        for intent in glob(os.path.join(LOCALE, "*", "*.intent")):
            for line in _read_lines(intent):
                for token in line.split():
                    if token.startswith("{") and token.endswith("}"):
                        self.assertEqual(
                            token, "{query}",
                            f"unexpected slot {token} in {intent}",
                        )

    def test_resource_base_names_are_intent2_compliant(self):
        """OVOS-INTENT-2 §2: intent/dialog resource base names must be
        lowercase ASCII letters, digits and underscores only (no dots, no
        mixed case). This governs ``.intent``/``.dialog`` files, which the
        rename in this repo's history actually normalized; ``.voc`` keyword
        lists are excluded since some locales carry legacy proper-noun
        vocabularies (e.g. subreddit names) that are not authoring-format
        violations."""
        pattern = re.compile(r"^[a-z0-9_]+$")
        for ext in ("intent", "dialog"):
            for path in glob(os.path.join(LOCALE, "*", f"*.{ext}")):
                base = os.path.splitext(os.path.basename(path))[0]
                self.assertRegex(
                    base, pattern,
                    f"{path} has a non-compliant base name {base!r}",
                )

    def test_no_duplicate_lowercase_locale_dirs(self):
        """Locale dirs must be canonical BCP-47 (fr-FR, not fr-fr)."""
        dirs = [d for d in os.listdir(LOCALE)
                if os.path.isdir(os.path.join(LOCALE, d))]
        seen = {}
        for d in dirs:
            key = d.lower()
            self.assertNotIn(
                key, seen, f"duplicate locale dir {d} vs {seen.get(key)}")
            seen[key] = d


class TestQueryBlacklist(TestCase):
    """The blacklist makes anaphoric fillers leave {query} unresolved."""

    def setUp(self):
        self.blacklist = _read_lines(os.path.join(EN_US, "query.blacklist"))

    def test_anaphora_are_blacklisted(self):
        for filler in ("that", "this", "it", "that one"):
            self.assertIn(filler, self.blacklist)

    def test_set_the_wallpaper_to_that_query_is_blacklisted(self):
        """'set the wallpaper to that' -> query='that' is rejected, not a topic."""
        query = "that"
        self.assertIn(query, self.blacklist)

    def test_real_topics_not_blacklisted(self):
        for topic in ("cats", "space", "mountains"):
            self.assertNotIn(topic, self.blacklist)
