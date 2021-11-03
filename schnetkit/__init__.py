from specable import Interface

from .schnet import SchNet
from .engine import load, Model
from .ase import Calculator, Converter

models = [SchNet]

loader = Interface("schnetkit", "models")
from_dict = loader.from_dict
from_yaml = loader.from_yaml
to_yaml = loader.to_yaml
