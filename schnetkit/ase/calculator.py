import torch
from torch.autograd import grad

from ase.calculators.calculator import Calculator as aseCalculator
from ase.constraints import full_3x3_to_voigt_6_stress

from schnetkit.helpers import guess_device_settings
from schnetkit import Model, keys

from .converter import Converter


class Calculator(aseCalculator):
    implemented_properties = [
        "energy",
        "forces",
        "stress",
        "energies",
    ]

    def __init__(
        self,
        model,
        device=None,
        stress=False,
        energies=False,
        skin=0.0,
        **kwargs,
    ):
        aseCalculator.__init__(self, **kwargs)

        if not isinstance(model, Model):
            from schnetkit import load

            model = load(model)

        self.model = model
        self.model.training = False  # avoid building graph for second-order derivatives

        self.device, _ = guess_device_settings(device=device)
        self.model.to(self.device)

        self.energies = energies
        self.stress = stress

        self.converter = Converter(
            cutoff=model.config_representation["cutoff"], device=self.device, skin=skin
        )

    @property
    def stress(self):
        return self._stress

    @stress.setter
    def stress(self, compute_stress):
        if compute_stress:
            self.model.stress = True
            self._stress = True
        else:
            self.model.stress = False
            self._stress = False

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=None,
        **kwargs,
    ):
        aseCalculator.calculate(self, atoms)  # sets self.atoms, which is ignored

        converted = self.converter(atoms)
        model_results = self.model(converted)

        results = {}

        energy = model_results[keys.energy].detach().cpu().numpy()
        results["energy"] = energy.item()

        forces = model_results[keys.forces].detach().cpu().numpy()
        results["forces"] = forces.reshape((len(atoms), 3))

        if self.stress:
            stress = model_results[keys.stress].detach().cpu().numpy().reshape((3, 3))
            results["stress"] = full_3x3_to_voigt_6_stress(stress)

        if self.energies:
            energies = model_results[keys.energies].detach().cpu().numpy()
            results["energies"] = energies.reshape(len(atoms))

        self.results = results

        return results
