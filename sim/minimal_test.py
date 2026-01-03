from datetime import datetime

from core.engine import DCUEngine
from core.claims.claim import Claim
from core.signals.signal import Signal

if __name__ == "__main__":
    engine = DCUEngine()
    claim = Claim(id="example", description="Test Claim", namespace="demo", created_at=datetime.now())
    engine.define_claim(claim)

    signal1 = Signal(claim_id="example", polarity=1, strength=0.5, source_id="s1", source_trust=1.0, timestamp=datetime.now())
    engine.add_signal(signal1)

    print(engine.get_confidence("example"))
