import unittest
from datetime import datetime

from core.engine import DCUEngine


class TestMergeSignalEvents(unittest.TestCase):
    def test_same_signal_id_ingested_twice_does_not_double_count_after_merge(self):
        t0 = datetime(2020, 1, 1, 0, 0, 0)

        a = DCUEngine(replica_id="A", half_life_seconds=10_000)
        b = DCUEngine(replica_id="B", half_life_seconds=10_000)

        a.define_claim("c1", description="Test", created_at=t0)
        b.define_claim("c1", description="Test", created_at=t0)

        # Duplicate delivery of the same globally-unique event.
        a.add_signal(
            claim_id="c1",
            polarity=+1,
            strength=1.0,
            source_id="s1",
            source_trust=1.0,
            timestamp=t0,
            signal_id="global-evt-1",
        )
        b.add_signal(
            claim_id="c1",
            polarity=+1,
            strength=1.0,
            source_id="s1",
            source_trust=1.0,
            timestamp=t0,
            signal_id="global-evt-1",
        )

        c_a = a.get_confidence("c1")

        # After merge, confidence should remain the same (not double-counted).
        a.merge_snapshot(b.export_snapshot(now=t0), now=t0, strategy="conservative_max")
        c_merged = a.get_confidence("c1")

        self.assertEqual(c_a, c_merged)


if __name__ == "__main__":
    unittest.main()
