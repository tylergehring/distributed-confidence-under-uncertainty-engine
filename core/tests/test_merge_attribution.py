import unittest
from datetime import datetime

from core.engine import DCUEngine


class TestMergeAttribution(unittest.TestCase):
    def test_attribution_merge_and_tombstones_prevent_double_apply(self):
        t0 = datetime(2020, 1, 1, 0, 0, 0)

        a = DCUEngine(replica_id="A")
        b = DCUEngine(replica_id="B")

        a.define_claim("c1", description="Test", created_at=t0)
        b.define_claim("c1", description="Test", created_at=t0)

        # b observes signals (no explicit source_trust required)
        b.add_signal(claim_id="c1", polarity=+1, strength=1.0, source_id="s1", timestamp=t0, signal_id="evt-1")
        b.add_signal(claim_id="c1", polarity=+1, strength=1.0, source_id="s1", timestamp=t0, signal_id="evt-2")

        # a merges b's snapshot and applies outcome once
        a.merge_snapshot(b.export_snapshot(now=t0), now=t0, strategy="conservative_max")
        a.apply_outcome("c1", outcome=True, now=t0)

        trust_after_a = a.get_source_trust("s1")
        self.assertTrue(trust_after_a > 0.5)

        # a shares back; b should learn the trust via merge
        b.merge_snapshot(a.export_snapshot(now=t0), now=t0, strategy="conservative_max")
        trust_after_b = b.get_source_trust("s1")
        self.assertEqual(trust_after_a, trust_after_b)

        # Applying outcome on b should NOT change trust (tombstones prevent replay)
        b.apply_outcome("c1", outcome=True, now=t0)
        self.assertEqual(trust_after_b, b.get_source_trust("s1"))


if __name__ == "__main__":
    unittest.main()
