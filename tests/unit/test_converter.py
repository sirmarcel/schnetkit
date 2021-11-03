from unittest import TestCase


class TestConverter(TestCase):
    def test_basic(self):
        from ase.build import bulk

        atoms = bulk("Ar", cubic=True) * [5, 5, 5]

        from schnetkit import Converter

        converter = Converter(cutoff=5.0, skin=0.5)

        converted = converter(atoms)

        assert not converter.needs_update(atoms)

        updated = atoms.get_positions()
        updated[0, 0] += 0.2
        atoms.set_positions(updated)
        assert not converter.needs_update(atoms)

        updated = atoms.get_positions()
        updated[0, 0] += 0.6
        atoms.set_positions(updated)
        assert converter.needs_update(atoms)

        assert converter.needs_update(atoms * [2, 2, 2])
