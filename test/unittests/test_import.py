"""Smoke import test for the wallpapers skill package."""
from unittest import TestCase


class TestImport(TestCase):
    def test_skill_class_importable(self):
        import ovos_skill_wallpapers
        self.assertTrue(hasattr(ovos_skill_wallpapers, "WallpapersSkill"))
