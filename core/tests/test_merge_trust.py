import unittest
from datetime import datetime

from core.engine import DCUEngine


class TestMergeTrust(unittest.TestCase):
    def test_trust_merges_idempotently_conservative_max(self):
        t0 = datetime(2020, 1, 1, 0, 0, 0)

        a = DCUEngine(replica_id="A")
        b = DCUEngine(replica_id="A")

        a.define_claim("c1", description="Test", created_at=t0)
        b.define_claim("c1", description="Test", created_at=t0)

        # Same replica_id implies overlapping trust evidence.
        a.add_signal(claim_id="c1", polarity=+1, strength=1.0, source_id="s1", timestamp=t0, signal_id="x1")
        b.add_signal(claim_id="c1", polarity=+1, strength=1.0, source_id="s1", timestamp=t0, signal_id="x2")

        a.apply_outcome("c1", outcome=True, now=t0)
        b.apply_outcome("c1", outcome=True, now=t0)

        trust_before = a.get_source_trust("s1")

        snap_b = b.export_snapshot(now=t0)
        a.merge_snapshot(snap_b, now=t0, strategy="conservative_max")
        trust_after = a.get_source_trust("s1")

        # Because both represent the same replica's evidence, merging should not inflate trust.
        self.assertEqual(trust_before, trust_after)

        # Merging again should still not change.
        a.merge_snapshot(snap_b, now=t0, strategy="conservative_max")
        self.assertEqual(trust_after, a.get_source_trust("s1"))

    def test_trust_additive_sum_can_accumulate_disjoint_replicas(self):
        t0 = datetime(2020, 1, 1, 0, 0, 0)

        a = DCUEngine(replica_id="A")
        b = DCUEngine(replica_id="B")

        a.define_claim("c1", description="Test", created_at=t0)
        b.define_claim("c1", description="Test", created_at=t0)

        a.add_signal(claim_id="c1", polarity=+1, strength=1.0, source_id="s1", timestamp=t0, signal_id="a1")
        b.add_signal(claim_id="c1", polarity=+1, strength=1.0, source_id="s1", timestamp=t0, signal_id="b1")

        a.apply_outcome("c1", outcome=True, now=t0)
        b.apply_outcome("c1", outcome=True, now=t0)

        trust_before = a.get_source_trust("s1")
        a.merge_snapshot(b.export_snapshot(now=t0), now=t0, strategy="additive_sum")
        trust_after = a.get_source_trust("s1")

        self.assertTrue(trust_after >= trust_before)


if __name__ == "__main__":
    unittest.main()
