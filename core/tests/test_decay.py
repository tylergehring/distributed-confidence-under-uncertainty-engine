from datetime import timedelta
from dcue_engine.core.decay.decay_function import exponential_decay

def test_decay():
    v = 1.0
    elapsed = 60  # seconds
    half_life = 60
    decayed = exponential_decay(v, elapsed, half_life)
    assert 0.49 < decayed < 0.51
