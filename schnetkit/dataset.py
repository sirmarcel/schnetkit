import numpy as np
from functools import partial
from pathlib import Path

import torch
from ase.calculators.calculator import PropertyNotImplementedError

from schnetpack.environment import AseEnvironmentProvider
from schnetpack.data.atoms import _convert_atoms

# TODO: replace with comms when we decide to kill legacy support
from vibes.helpers import progressbar

from schnetkit import keys
from .engine import Stateful

nan = float("nan")


class Dataset(torch.utils.data.Dataset, Stateful):
    """Dataset.

    Replacement for `AtomsData` in SchNetPack, dropping non-essential
    features for speed and simplicity at the expense of generality.

    torch `Datasets` represents a "database" of samples that can be queried by
    index using the `__getitem__` method. For training, they're collated into
    batches by a `DataLoader`. SchNetPack models expect a dictionary as input,
    with the various entries corresponding to `torch` tensors with positions,
    neighbor indices, and so on. Converting an `ase.Atoms` object, or other
    general representation of a structure, into this format can be expensive, as
    it requires computing neighbourlists.

    Here, we therefore do this computation *ahead of time*, store the resulting
    list of dictionaries on disk, and load it into RAM *once*. This assumes that
    the whole dataset fits into memory, which is a fair assumption for DFT datasets,
    and in exchange, allows for much less overhead, and minimal i/o trouble.

    The `Stateful` mixin takes care of de-serialisation (we simply store everything
    as a `torch`-style pickle), so we can focus on the conversion aspect.

    In general, this class works as follows: We initialise with some general settings,
    mainly the cutoff to employ for neighborlist generation, and receive an empty dataset
    in return. Then we can add `Atoms` objects to it one by one. These `Atoms` are
    converted and then stored in order in a list. Any sorting or sampling has to
    occur at conversion time, or in the `DataLoader` later.

    We currently support only `energy`, `forces` and `stress` as labels. `stress` is
    allowed to only be present in some samples, it will be given as `NaN`.

    Attributes:
        cutoff: float indicating cutoff of neighborlists
                (should be >= the one in the model you're training)
        offset: float to subtract from potential energy
                (chould be the mean over your dataset)

    """

    def __init__(self, cutoff=5.0, offset=0.0):
        self.offset = offset
        self.cutoff = cutoff

        self.converter = partial(
            _convert_atoms,
            environment_provider=AseEnvironmentProvider(cutoff=cutoff),
        )

        self.data = []

    # torch interface

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx]

    def __len__(self):
        return len(self.data)

    # de/serialisation

    def get_dict(self):
        return {"offset": self.offset, "cutoff": self.cutoff}

    def get_state(self):
        return {"data": self.data}

    def restore(self, state):
        self.data = state["data"]

    # adding atoms

    def add_atoms(self, iterable, stress=True):
        """Add iterable of Atoms.

        Expecting a `SinglePointCalculator` to be attached
        to each `Atoms` object, providing at least `energy`
        and `forces`.

        Args:
            iterable: something that iterates over `ase.Atoms`
                (or a single `Atoms`, for convenience)
            stress: boolean, if True will try to add stress, and
                substitute NaNs if not present

        """
        from ase import Atoms

        if isinstance(iterable, Atoms):
            iterable = [Atoms]

        for atoms in progressbar(iterable, prefix="importing atoms..."):
            self.data.append(
                convert_atoms(atoms, self.converter, self.offset, stress=stress)
            )

    def add_file(self, file, stress=True):
        """Add trajectory of Atoms.

        Like add_atoms, but reading `vibes`-style `.son` and `.nc` files,
        as well as `ase` `.traj` files.

        """

        file = Path(file)
        suffix = file.suffix

        if suffix == ".son" or suffix == ".nc":
            from vibes.trajectory import reader

            self.add_atoms(reader(file), stress=stress)
        elif suffix == ".traj":
            from ase.io import Trajectory

            self.add_atoms(Trajectory(file), stress=stress)

    # convenience for inspection

    @property
    def energy(self):
        return torch.tensor([data[keys.energy] for data in self.data]).squeeze()

    @property
    def forces(self):
        # can't meaningfully cast to tensor,
        # as forces might be ragged
        return [data[keys.forces] for data in self.data]

    @property
    def stress(self):
        return torch.tensor([data[keys.stress].tolist() for data in self.data])


def convert_atoms(atoms, converter, offset, stress=True):
    dictionary = converter(atoms)  # neighborlist, etc.

    energy = atoms.get_potential_energy() - offset
    dictionary[keys.energy] = torch.tensor(energy, dtype=torch.float).unsqueeze(0)
    dictionary[keys.forces] = torch.tensor(atoms.get_forces(), dtype=torch.float)

    if stress:
        try:
            dictionary[keys.stress] = torch.tensor(
                atoms.get_stress(voigt=False), dtype=torch.float
            )
        except PropertyNotImplementedError:
            dictionary[keys.stress] = torch.tensor(
                nan * np.zeros((3, 3)), dtype=torch.float
            )

    return torchify_dict(dictionary)


def torchify_dict(data):
    """Transform np.ndarrays to torch.tensors in dicts.

    (Borrowed from SchNetPack)
    """

    torchified = {}
    for name, prop in data.items():

        if prop.dtype in [np.int, np.int32, np.int64]:
            torchified[name] = torch.LongTensor(prop)
        elif prop.dtype in [np.float, np.float32, np.float64]:
            torchified[name] = torch.FloatTensor(prop.copy())
        elif torch.is_tensor(prop):
            torchified[name] = prop
        else:
            raise RuntimeError(f"Invalid datatype {type(prop)} for property {name}!")
    return torchified
