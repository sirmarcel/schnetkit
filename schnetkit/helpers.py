import torch


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
