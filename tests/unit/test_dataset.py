import numpy as np

from unittest import TestCase
from ase.io import Trajectory

import schnetkit
from schnetkit.dataset import Dataset


from tempdir import Tempdir

example = "../assets/md.traj"


def compare_datasets(a, b):
    for i, entry in enumerate(a):
        for key in entry.keys():
            x = entry[key].numpy()
            y = b[i][key].numpy()
            np.testing.assert_allclose(x, y)


class TestDataset(Tempdir, TestCase):
    def test_roundtrip(self):
        data = Dataset()
        data.add_atoms(iterable=Trajectory(example), stress=True)
        data.save(self.tempdir / "data")
        data2 = schnetkit.load(self.tempdir / "data")

        compare_datasets(data, data2)

    def test_add_file(self):
        data = Dataset()
        data.add_atoms(iterable=Trajectory(example), stress=True)
        data2 = Dataset()
        data2.add_file(example, stress=True)

        compare_datasets(data, data2)

        for i, atoms in enumerate(Trajectory(example)):
            np.testing.assert_allclose(
                data[i]["energy"].numpy(), atoms.get_potential_energy()
            )
            np.testing.assert_allclose(data[i]["forces"].numpy(), atoms.get_forces())
            np.testing.assert_allclose(
                data[i]["stress"].numpy(), atoms.get_stress(voigt=False)
            )
