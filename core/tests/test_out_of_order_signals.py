import unittest
from datetime import datetime

from core.engine import DCUEngine


class TestOutOfOrderSignals(unittest.TestCase):
    def test_out_of_order_events_match_in_order_result(self):
        t0 = datetime(2020, 1, 1, 0, 0, 0)
        t1 = datetime(2020, 1, 1, 0, 10, 0)

        a = DCUEngine(replica_id="A", half_life_seconds=10_000)
        b = DCUEngine(replica_id="B", half_life_seconds=10_000)

        a.define_claim("c1", description="Test", created_at=t0)
        b.define_claim("c1", description="Test", created_at=t0)

        # In-order ingestion
        a.add_signal(
            claim_id="c1",
            polarity=+1,
            strength=1.0,
            source_id="s1",
            source_trust=1.0,
            timestamp=t0,
            signal_id="evt-early",
        )
        a.add_signal(
            claim_id="c1",
            polarity=+1,
            strength=1.0,
            source_id="s1",
            source_trust=1.0,
            timestamp=t1,
            signal_id="evt-late",
        )

        # Out-of-order ingestion
        b.add_signal(
            claim_id="c1",
            polarity=+1,
            strength=1.0,
            source_id="s1",
            source_trust=1.0,
            timestamp=t1,
            signal_id="evt-late",
        )
        b.add_signal(
            claim_id="c1",
            polarity=+1,
            strength=1.0,
            source_id="s1",
            source_trust=1.0,
            timestamp=t0,
            signal_id="evt-early",
        )

        self.assertEqual(a.get_confidence("c1"), b.get_confidence("c1"))


if __name__ == "__main__":
    unittest.main()
