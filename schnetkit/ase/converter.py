import torch
from collections import namedtuple
from schnetpack.data.atoms import AtomsConverter
from schnetpack.environment import AseEnvironmentProvider, BaseEnvironmentProvider

from schnetkit.helpers import guess_device_settings


Converted = namedtuple("Converted", ["inputs", "neighborhood"])
Neighborhood = namedtuple("Neighborhood", ["idx", "offset"])


class Converter:
    """Convert atoms to SchNetPack format.

    This is a thin layer on top of the SchNetPack methods to transform
    `ase.Atoms` objects into a suitable format for SchNet. This implementation
    adds the common performance-enhancing practice of not re-computing
    neighborhoods at every step, but only if atoms move more than the `skin`
    parameter. (Or when the periodic boundary conditions change.)

    NOTE: `skin` defaults to 0.0 for compatibility with legacy infrastructure,
        to see performance benefits it needs to be set to something >0.0.

    NOTE: `skin` fails if the cell changes every step, so it won't be fast for NPT.

    """

    def __init__(self, cutoff, device=None, skin=0.0):
        self.device, _ = guess_device_settings(device=device)

        self.skin = skin
        self.cutoff = cutoff + skin

        self.environment_provider = AseEnvironmentProvider(cutoff=self.cutoff)

        self.atoms = None
        self.neighborhood = None

    def __call__(self, atoms):
        if self.needs_update(atoms):
            self.atoms = atoms.copy()
            self.neighborhood = self.get_neighborhood(self.atoms)

        converter = AtomsConverter(
            environment_provider=FakeEnvironmentProvider(self.neighborhood),
            device=self.device,
        )
        inputs = converter(atoms)

        return Converted(inputs, self.neighborhood)

    def get_neighborhood(self, atoms):
        idx, offset = self.environment_provider.get_environment(atoms)

        return Neighborhood(idx, offset)

    def needs_update(self, atoms):
        if self.atoms is None:
            self.atoms = atoms
            return True
        else:
            return self._needs_update(atoms)

    def _needs_update(self, atoms):
        if (
            (self.atoms.get_cell() != atoms.get_cell()).any()
            or (self.atoms.get_pbc() != atoms.get_pbc()).any()
            or self.atoms.get_positions().shape != atoms.get_positions().shape
            or (self.atoms.get_atomic_numbers() != atoms.get_atomic_numbers()).any()
        ):
            return True

        my_positions = self.atoms.get_positions()
        new_positions = atoms.get_positions()
        return ((my_positions - new_positions) ** 2).sum(1).max() > (
            0.5 * self.skin
        ) ** 2


class FakeEnvironmentProvider(BaseEnvironmentProvider):
    def __init__(self, neighborhood):
        self.neighborhood = neighborhood

    def get_environment(self, atoms):
        return self.neighborhood.idx, self.neighborhood.offset
