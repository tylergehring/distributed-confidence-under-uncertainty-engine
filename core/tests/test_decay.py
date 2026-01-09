from core.decay.decay_function import exponential_decay

import unittest


class TestDecay(unittest.TestCase):
    def test_decay(self):
        v = 1.0
        elapsed = 60  # seconds
        half_life = 60
        decayed = exponential_decay(v, elapsed, half_life)
        self.assertTrue(0.49 < decayed < 0.51)


if __name__ == "__main__":
    unittest.main()
