import torch


# TODO: replace with comms
def talk(msg, **kwargs):
    """wrapper for `utils.talk` with prefix"""
    from vibes.helpers import talk as _talk
    return _talk(msg, prefix="schnetkit", **kwargs)


def guess_device_settings(device=None, parallel=None):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    if parallel is None:
        if torch.cuda.device_count() > 1 and device == "cuda":
            parallel = True
        else:
            parallel = False

    return device, parallel
