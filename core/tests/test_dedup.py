import unittest
from datetime import datetime, timedelta

from core.engine import DCUEngine


class TestDedup(unittest.TestCase):
    def test_duplicate_signal_id_is_ignored(self):
        t0 = datetime(2020, 1, 1, 0, 0, 0)
        engine = DCUEngine(half_life_seconds=10_000, dedup_window_seconds=3600, dedup_max_per_claim=100)
        engine.define_claim("c1", description="Test", created_at=t0)

        engine.add_signal(
            claim_id="c1",
            polarity=+1,
            strength=1.0,
            source_id="s1",
            source_trust=1.0,
            timestamp=t0,
            signal_id="evt-1",
        )
        c1 = engine.get_confidence("c1")

        # Same signal replayed should not change confidence.
        engine.add_signal(
            claim_id="c1",
            polarity=+1,
            strength=1.0,
            source_id="s1",
            source_trust=1.0,
            timestamp=t0,
            signal_id="evt-1",
        )
        c2 = engine.get_confidence("c1")
        self.assertEqual(c1, c2)

    def test_dedup_ttl_allows_eventual_reuse(self):
        t0 = datetime(2020, 1, 1, 0, 0, 0)
        engine = DCUEngine(half_life_seconds=10_000, dedup_window_seconds=10, dedup_max_per_claim=100)
        engine.define_claim("c1", description="Test", created_at=t0)

        engine.add_signal(
            claim_id="c1",
            polarity=+1,
            strength=1.0,
            source_id="s1",
            source_trust=1.0,
            timestamp=t0,
            signal_id="evt-ttl",
        )
        c1 = engine.get_confidence("c1")

        # After TTL has expired, bounded-memory dedup may forget old IDs.
        # However, signal_id is treated as globally unique; do not reuse it for a different event.
        t_late = t0 + timedelta(seconds=11)
        engine.add_signal(
            claim_id="c1",
            polarity=+1,
            strength=1.0,
            source_id="s1",
            source_trust=1.0,
            timestamp=t_late,
            signal_id="evt-ttl-2",
        )
        c2 = engine.get_confidence("c1")
        self.assertTrue(c2 > c1)

    def test_dedup_capacity_can_forget_old_ids(self):
        t0 = datetime(2020, 1, 1, 0, 0, 0)
        engine = DCUEngine(half_life_seconds=10_000, dedup_window_seconds=3600, dedup_max_per_claim=1)
        engine.define_claim("c1", description="Test", created_at=t0)

        engine.add_signal(
            claim_id="c1",
            polarity=+1,
            strength=1.0,
            source_id="s1",
            source_trust=1.0,
            timestamp=t0,
            signal_id="evt-a",
        )
        c1 = engine.get_confidence("c1")

        engine.add_signal(
            claim_id="c1",
            polarity=+1,
            strength=1.0,
            source_id="s1",
            source_trust=1.0,
            timestamp=t0,
            signal_id="evt-b",
        )

        # With capacity=1, evt-a was evicted; replaying it can be counted again.
        engine.add_signal(
            claim_id="c1",
            polarity=+1,
            strength=1.0,
            source_id="s1",
            source_trust=1.0,
            timestamp=t0,
            signal_id="evt-a",
        )
        c2 = engine.get_confidence("c1")
        self.assertTrue(c2 > c1)


if __name__ == "__main__":
    unittest.main()
