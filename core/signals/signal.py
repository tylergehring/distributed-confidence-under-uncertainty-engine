from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class Signal:
    claim_id: str
    polarity: float
    strength: float
    source_id: str
    timestamp: datetime
    source_trust: Optional[float] = None
    signal_id: Optional[str] = None
    