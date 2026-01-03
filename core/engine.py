from datetime import datetime

from .claims.claim import Claim
from .signals.signal import Signal
from .confidence.confidence_state import ConfidenceState
from .decay.decay_function import exponential_decay


class DCUEngine:
    def __init__(self, half_life_seconds=1600):
        self.claims = {}
        self.confidence = {}
        self.half_life_seconds = half_life_seconds
        
        
    def define_claim(self, claim: Claim):
        self.claims[claim.id] = claim
        self.confidence[claim.id] = ConfidenceState(
            claim_id= claim.id, value= 0.0, lower_bound= 0.0, upper_bound= 1.0, last_updated=datetime.now()
            )
        
    def add_signal(self, signal: Signal):
        #simple additive model for now
        cs = self.confidence[signal.claim_id]
        elapsed = (signal.timestamp - cs.last_updated).total_seconds()
        cs.value = exponential_decay(cs.value, elapsed, self.half_life_seconds)
        
        #update confidence with new signal weighted by trust
        cs.value += signal.polarity * signal.strength * signal.source_trust
        cs.last_updated = signal.timestamp
        self.confidence[signal.claim_id] = cs
        
    def get_confidence(self, claim_id: str):
        return self.confidence[claim_id].value 