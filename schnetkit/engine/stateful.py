"""stateful.py

Implements basic infrastructure for saving and
loading models. We decompose into two parts:

A `spec`, which describes model architecture as
a `dict`, and `state`, which is the `state_dict`
we get from `torch`, which holds all the weights.

Together, this allows us to reconstruct models fully.

"""


import torch
from specable import Specable

from .legacy import from_gknet


def load(path):

    if isinstance(path, Stateful):
        return path
    else:
        spec, state = load_file(path)

        from schnetkit import from_dict

        obj = from_dict(spec)
        obj.restore(state)

        return obj


def load_file(path):
    payload = torch.load(path)

    # deal with outdated formats
    from_gknet(payload)

    spec = payload["spec"]
    state = payload["state"]

    return spec, state


class Stateful(Specable):
    def get_state(self):
        raise NotImplementedError

    def restore(self, state):
        raise NotImplementedError

    def save(self, path):
        payload = {"state": self.get_state(), "spec": self.to_dict()}
        torch.save(payload, path)
