import numpy as np
from unittest import TestCase


class TestLegacy(TestCase):
    def setUp(self):
        from schnetkit import load

        model_old = load("model_legacy.torch")
        model_old.save("model.torch")
        model_new = load("model.torch")

        from ase.io import read

        self.atoms = read("geometry.in")

    def test_calclator(self):

        from schnetkit import Calculator, keys

        calc_new = Calculator("model.torch", stress=True, energies=True)
        calc_old = Calculator("model_legacy.torch", stress=True, energies=True)

        pred_new = calc_new.calculate(self.atoms)
        pred_old = calc_old.calculate(self.atoms)

        self.assertTrue(keys.energy in pred_old)
        self.assertTrue(keys.forces in pred_old)
        self.assertTrue(keys.stress in pred_old)

        for key in pred_old:
            np.testing.assert_allclose(pred_old[key], pred_new[key])
