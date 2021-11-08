from unittest import TestCase

from tempdir import Tempdir


class TestSchnet(Tempdir, TestCase):
    def test_fire(self):
        from schnetkit import SchNet

        SchNet()

    def test_yaml(self):
        from schnetkit import SchNet, from_yaml, to_yaml

        schnet = SchNet(representation={"cutoff": 6.0}, atomwise={"n_layers": 3})
        to_yaml(self.tempdir / "model.yaml", schnet)
        schnet2 = from_yaml(self.tempdir / "model.yaml")

        self.assertEqual(
            schnet.cutoff,
            schnet2.cutoff,
        )
