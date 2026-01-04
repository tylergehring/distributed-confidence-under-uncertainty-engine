import unittest
from datetime import datetime

from core.engine import DCUEngine


class TestMergeSnapshot(unittest.TestCase):
    def test_merge_conservative_max(self):
        t0 = datetime(2020, 1, 1, 0, 0, 0)
        now = datetime(2020, 1, 1, 0, 10, 0)

        a = DCUEngine(half_life_seconds=10_000, replica_id="A")
        b = DCUEngine(half_life_seconds=10_000, replica_id="A")

        a.define_claim("c1", description="Test", created_at=t0)
        b.define_claim("c1", description="Test", created_at=t0)

        a.add_signal(
            claim_id="c1",
            polarity=+1,
            strength=2.0,
            source_id="s1",
            source_trust=1.0,
            timestamp=t0,
            signal_id="evt-dup",
        )

        # b sees the same evidence (e.g., duplicated delivery)
        b.add_signal(
            claim_id="c1",
            polarity=+1,
            strength=2.0,
            source_id="s1",
            source_trust=1.0,
            timestamp=t0,
            signal_id="evt-dup",
        )

        snap_b = b.export_snapshot(now=now)
        a.merge_snapshot(snap_b, now=now, strategy="conservative_max")

        # Conservative merge should not double-count the duplicated evidence.
        c = a.get_confidence("c1")
        self.assertTrue(0.7 < c < 0.9)

    def test_merge_additive_sum_increases(self):
        t0 = datetime(2020, 1, 1, 0, 0, 0)
        now = datetime(2020, 1, 1, 0, 10, 0)

        a = DCUEngine(half_life_seconds=10_000, replica_id="A")
        b = DCUEngine(half_life_seconds=10_000, replica_id="B")

        a.define_claim("c1", description="Test", created_at=t0)
        b.define_claim("c1", description="Test", created_at=t0)

        a.add_signal(
            claim_id="c1",
            polarity=+1,
            strength=2.0,
            source_id="s1",
            source_trust=1.0,
            timestamp=t0,
            signal_id="evt-a",
        )

        # b sees different evidence
        b.add_signal(
            claim_id="c1",
            polarity=+1,
            strength=2.0,
            source_id="s2",
            source_trust=1.0,
            timestamp=t0,
            signal_id="evt-b",
        )

        c_before = a.get_confidence("c1")
        snap_b = b.export_snapshot(now=now)
        a.merge_snapshot(snap_b, now=now, strategy="additive_sum")
        c_after = a.get_confidence("c1")

        self.assertTrue(c_after > c_before)


if __name__ == "__main__":
    unittest.main()
