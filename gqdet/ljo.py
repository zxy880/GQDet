"""Original LJO schedules. Epoch arguments use zero-based global epochs."""
import math

def lgrm_cost_class(global_epoch):
    """Classification matching-cost schedule in global epochs."""
    if global_epoch <= 54:
        return 2.0
    if global_epoch <= 69:
        progress = (global_epoch - 55) / 14.0
        return 1.0 + 0.5 * (1.0 + math.cos(math.pi * progress))
    return 1.0

def mal_gamma(global_epoch):
    """Only MAL's negative modulation changes; quality targets do not."""
    if global_epoch <= 54:
        return 1.5
    if global_epoch <= 79:
        progress = (global_epoch - 55) / 24.0
        return 2.0 - 0.25 * (1.0 + math.cos(math.pi * progress))
    return 2.0

def matching_class_cost(config, epoch, default):
    value = float(default)
    for item in sorted(config.get("matching_class_cost_overrides", []), key=lambda x: x["start_epoch"]):
        if epoch >= int(item["start_epoch"]):
            value = float(item["value"])
    return value
