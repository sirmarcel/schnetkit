from schnetkit.engine import Stateful


class Dummy(Stateful):

    def __init__(self, a=2):
        self.a = a

        self.state = "great"

    def get_dict(self):
        return {"a": self.a}

    def get_state(self):
        return {"state": self.state}

    def restore(self, payload):
        self.state = payload["state"]

    def work(self):
        self.state = "tired"


components = [Dummy]
