import math

def exponential_decay(value: float, elapsed_seconds: float, half_life_seconds: float) -> float:
    """ Decays the value towards 0 over time using a simple exponential decay"""
    decay_factor = 0.5 ** (elapsed_seconds / half_life_seconds)
    return value * decay_factor

