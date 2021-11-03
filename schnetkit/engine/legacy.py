def from_gknet(payload):

    if "config" in payload:
        spec = payload.pop("config")
        model = spec.pop("model")
        spec["schnet"] = model
        payload["spec"] = spec
