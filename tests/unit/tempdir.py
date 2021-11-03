import pathlib
import shutil


class Tempdir:
    def setUp(self):
        self.tempdir = (
            pathlib.Path(__file__) / ".."
        ).resolve() / f"tmp_{self.__class__.__name__}"
        self.tempdir.mkdir(exist_ok=True)

        if hasattr(self, "_setUp"):
            self._setUp()

    def tearDown(self):
        shutil.rmtree(self.tempdir)

        if hasattr(self, "_tearDown"):
            self._tearDown()
