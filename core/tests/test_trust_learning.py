import unittest
from datetime import datetime

from core.engine import DCUEngine


class TestTrustLearning(unittest.TestCase):
    def test_engine_uses_learned_trust_when_missing(self):
        t0 = datetime(2020, 1, 1, 0, 0, 0)
        engine = DCUEngine(half_life_seconds=10_000)
        engine.define_claim("c1", description="Test", created_at=t0)

        # With no history, trust defaults to 0.5; signal weight becomes 0.5.
        engine.add_signal(
            claim_id="c1",
            polarity=+1,
            strength=1.0,
            source_id="s1",
            timestamp=t0,
            # source_trust omitted
        )
        c = engine.get_confidence("c1")
        self.assertTrue(c > 0.5)

    def test_apply_outcome_updates_trust(self):
        t0 = datetime(2020, 1, 1, 0, 0, 0)
        engine = DCUEngine(half_life_seconds=10_000)
        engine.define_claim("c1", description="Test", created_at=t0)

        # Two sources give opposite stances.
        engine.add_signal(
            claim_id="c1",
            polarity=+1,
            strength=1.0,
            source_id="good",
            source_trust=0.5,
            timestamp=t0,
        )
        engine.add_signal(
            claim_id="c1",
            polarity=-1,
            strength=1.0,
            source_id="bad",
            source_trust=0.5,
            timestamp=t0,
        )

        engine.apply_outcome("c1", outcome=True)

        self.assertTrue(engine.get_source_trust("good") > 0.5)
        self.assertTrue(engine.get_source_trust("bad") < 0.5)

    def test_apply_outcome_uses_multiple_events_and_clears_by_default(self):
        t0 = datetime(2020, 1, 1, 0, 0, 0)
        engine = DCUEngine(
            half_life_seconds=10_000,
            outcome_attribution_window_seconds=3600,
            outcome_attribution_max_events=1000,
        )
        engine.define_claim("c1", description="Test", created_at=t0)

        # Same source sends two supporting signals.
        engine.add_signal(
            claim_id="c1",
            polarity=+1,
            strength=1.0,
            source_id="s1",
            source_trust=0.5,
            timestamp=t0,
            signal_id="evt-1",
        )
        engine.add_signal(
            claim_id="c1",
            polarity=+1,
            strength=1.0,
            source_id="s1",
            source_trust=0.5,
            timestamp=t0,
            signal_id="evt-2",
        )

        engine.apply_outcome("c1", outcome=True, now=t0)
        trust_after = engine.get_source_trust("s1")
        self.assertTrue(trust_after > 0.5)

        # Applying again should not change trust (events cleared by default).
        engine.apply_outcome("c1", outcome=True, now=t0)
        self.assertEqual(trust_after, engine.get_source_trust("s1"))


if __name__ == "__main__":
    unittest.main()
