from specable import Interface

from .schnet import SchNet
from .engine import load, Model
from .ase import Calculator, Converter
from .dataset import Dataset

components = [SchNet, Dataset]

loader = Interface("schnetkit", "components")
from_dict = loader.from_dict
from_yaml = loader.from_yaml
to_yaml = loader.to_yaml
