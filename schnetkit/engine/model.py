from .stateful import Stateful


class Model(Stateful):
    def restore(self, state):
        self.model.load_state_dict(state)

    def __call__(self, converted):
        return self.model(converted.inputs)
