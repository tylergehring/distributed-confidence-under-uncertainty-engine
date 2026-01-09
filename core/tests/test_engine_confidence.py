from datetime import datetime, timedelta

from core.engine import DCUEngine
import unittest


class TestEngineConfidence(unittest.TestCase):
    def test_confidence_is_bounded_and_updates(self):
        engine = DCUEngine(half_life_seconds=10_000)
        t0 = datetime(2020, 1, 1, 0, 0, 0)
        engine.define_claim("c1", description="Test", created_at=t0)
        engine.tick(t0)

        # Prior is ~0.5
        c0 = engine.get_confidence("c1")
        self.assertTrue(0.49 <= c0 <= 0.51)

        engine.add_signal(
            claim_id="c1",
            polarity=+1,
            strength=1.0,
            source_id="s1",
            source_trust=1.0,
            timestamp=t0,
        )
        c1 = engine.get_confidence("c1")
        self.assertTrue(0.5 < c1 <= 1.0)

        engine.add_signal(
            claim_id="c1",
            polarity=-1,
            strength=1.0,
            source_id="s2",
            source_trust=1.0,
            timestamp=t0,
        )
        c2 = engine.get_confidence("c1")
        self.assertTrue(0.0 <= c2 <= 1.0)

    def test_decay_moves_toward_prior(self):
        engine = DCUEngine(half_life_seconds=60)
        t0 = datetime(2020, 1, 1, 0, 0, 0)
        engine.define_claim("c1", description="Test", created_at=t0)
        engine.tick(t0)

        engine.add_signal(
            claim_id="c1",
            polarity=+1,
            strength=5.0,
            source_id="s1",
            source_trust=1.0,
            timestamp=t0,
        )

        c_strong = engine.get_confidence("c1")
        self.assertTrue(c_strong > 0.7)

        # After several half-lives, evidence should decay and confidence should move back toward 0.5.
        t_late = t0 + timedelta(seconds=60 * 10)
        c_late = engine.get_confidence("c1", now=t_late)
        self.assertTrue(0.49 <= c_late <= 0.60)

    def test_unknown_claim_raises(self):
        engine = DCUEngine()
        with self.assertRaises(KeyError):
            engine.get_confidence("missing")


if __name__ == "__main__":
    unittest.main()
