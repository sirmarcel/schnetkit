from unittest import TestCase

from pathlib import Path
import sys

from tempdir import Tempdir

sys.path.append(Path() / "dummy")


class TestStateful(Tempdir, TestCase):
    def test_rountrip(self):
        import schnetkit
        from dummy import Dummy

        dummy = Dummy(a=1)
        dummy.work()
        dummy.save(self.tempdir / "dummy")
        dummy2 = schnetkit.load(self.tempdir / "dummy")

        self.assertEqual(dummy2.state, dummy.state)
        self.assertEqual(dummy2.a, dummy.a)
