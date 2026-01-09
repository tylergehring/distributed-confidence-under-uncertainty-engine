from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

from .claims.claim import Claim
from .signals.signal import Signal
from .confidence.confidence_state import ConfidenceState
from .decay.decay_function import exponential_decay
from .merge.merge_state import MergeStrategy, merge_confidence_snapshots, snapshot_from_state
from .merge.merge_trust import merge_trust_snapshots, snapshot_from_trust_state
from .trust.source_trust import SourceTrustState


class DCUEngine:
    def __init__(
        self,
        half_life_seconds=1600,
        replica_id: str = "local",
        *,
        dedup_window_seconds: float = 3600.0,
        dedup_max_per_claim: int = 10_000,
        signal_event_window_seconds: float = 24 * 3600.0,
        signal_event_max_events: int = 50_000,
        outcome_attribution_window_seconds: float = 24 * 3600.0,
        outcome_attribution_max_events: int = 50_000,
    ):
        if half_life_seconds <= 0:
            raise ValueError("half_life_seconds must be > 0")
        if not replica_id:
            raise ValueError("replica_id must be a non-empty string")
        if dedup_window_seconds < 0:
            raise ValueError("dedup_window_seconds must be >= 0")
        if dedup_max_per_claim < 0:
            raise ValueError("dedup_max_per_claim must be >= 0")
        if signal_event_window_seconds < 0:
            raise ValueError("signal_event_window_seconds must be >= 0")
        if signal_event_max_events < 0:
            raise ValueError("signal_event_max_events must be >= 0")
        if outcome_attribution_window_seconds < 0:
            raise ValueError("outcome_attribution_window_seconds must be >= 0")
        if outcome_attribution_max_events < 0:
            raise ValueError("outcome_attribution_max_events must be >= 0")
        self.claims = {}
        self.confidence = {}
        self.half_life_seconds = half_life_seconds
        self.replica_id = replica_id

        # Learned trust per source (optional; used when a Signal omits source_trust).
        self.source_trust: Dict[str, SourceTrustState] = {}
        # Minimal attribution memory for outcome feedback: claim_id -> source_id -> (polarity, weight, timestamp)
        self._last_source_stance_by_claim: Dict[str, Dict[str, Tuple[float, float, datetime]]] = {}

        # Bounded event log for outcome attribution: claim_id -> list[(source_id, polarity, strength, timestamp, signal_id)]
        self.outcome_attribution_window_seconds = float(outcome_attribution_window_seconds)
        self.outcome_attribution_max_events = int(outcome_attribution_max_events)

        # Mergeable outcome attribution events (OR-Set style with tombstones):
        # claim_id -> event_id -> (origin_replica_id, source_id, polarity, strength, timestamp)
        self._attribution_events_by_claim: Dict[str, Dict[str, Tuple[str, str, float, float, datetime]]] = {}
        # claim_id -> event_id -> applied_at
        self._attribution_applied_by_claim: Dict[str, Dict[str, datetime]] = {}
        self._attribution_event_counter = 0

        # Optional idempotency guard: per-claim map of signal_id -> last seen timestamp.
        # Tradeoff: bounded memory means very old IDs may be forgotten.
        self.dedup_window_seconds = float(dedup_window_seconds)
        self.dedup_max_per_claim = int(dedup_max_per_claim)
        self._seen_signal_ids_by_claim: Dict[str, Dict[str, datetime]] = {}

        # Mergeable signal event log (authoritative for confidence), keyed by globally-unique signal_id.
        # claim_id -> signal_id -> (source_id, polarity, strength, effective_trust, timestamp)
        self.signal_event_window_seconds = float(signal_event_window_seconds)
        self.signal_event_max_events = int(signal_event_max_events)
        self._signal_events_by_claim: Dict[str, Dict[str, Tuple[str, float, float, float, datetime]]] = {}
        self._signal_event_counter = 0
        # If a signal arrives with timestamp <= current last_updated, we need a full rebuild.
        self._signal_replay_dirty_by_claim: Dict[str, bool] = {}

        # Performance: cache a stable total order of signal_ids per claim and a replay index.
        # When in-order events are appended, we can apply only the tail without re-sorting or
        # scanning the whole dict.
        self._signal_event_order_by_claim: Dict[str, List[str]] = {}
        self._signal_event_replay_index_by_claim: Dict[str, int] = {}
        
    def _recompute_summary(self, cs: ConfidenceState, *, z: float = 1.96) -> None:
        # Keep alpha/beta consistent with per-replica evidence maps.
        support_total = sum(max(0.0, float(v)) for v in cs.support_evidence_by_replica.values())
        refute_total = sum(max(0.0, float(v)) for v in cs.refute_evidence_by_replica.values())
        cs.alpha = 1.0 + support_total
        cs.beta = 1.0 + refute_total

        # Mean and a simple normal-approx uncertainty band.
        denom = cs.alpha + cs.beta
        if denom <= 0:
            cs.value = 0.5
            cs.lower_bound = 0.0
            cs.upper_bound = 1.0
            return

        mean = cs.alpha / denom
        var = (cs.alpha * cs.beta) / ((denom**2) * (denom + 1.0))
        std = var ** 0.5

        cs.value = max(0.0, min(1.0, mean))
        cs.lower_bound = max(0.0, min(1.0, mean - z * std))
        cs.upper_bound = max(0.0, min(1.0, mean + z * std))

    def _apply_decay(self, cs: ConfidenceState, now: datetime) -> None:
        elapsed = (now - cs.last_updated).total_seconds()
        if elapsed < 0:
            raise ValueError("now must be >= last_updated")
        if elapsed == 0:
            return

        # Decay each replica's excess evidence toward 0 (the Beta(1,1) prior).
        for rid, value in list(cs.support_evidence_by_replica.items()):
            cs.support_evidence_by_replica[rid] = exponential_decay(
                max(0.0, float(value)),
                elapsed,
                self.half_life_seconds,
            )
        for rid, value in list(cs.refute_evidence_by_replica.items()):
            cs.refute_evidence_by_replica[rid] = exponential_decay(
                max(0.0, float(value)),
                elapsed,
                self.half_life_seconds,
            )

        cs.last_updated = now
        self._recompute_summary(cs)

    def define_claim(
        self,
        claim: Union[Claim, str],
        description: Optional[str] = None,
        namespace: str = "default",
        created_at: Optional[datetime] = None,
    ):
        if isinstance(claim, Claim):
            claim_obj = claim
        else:
            if description is None:
                raise ValueError("description is required when defining a claim by id")
            claim_obj = Claim(
                id=claim,
                description=description,
                namespace=namespace,
                created_at=created_at or datetime.now(),
            )

        self.claims[claim_obj.id] = claim_obj
        cs = ConfidenceState(
            claim_id=claim_obj.id,
            value=0.5,
            lower_bound=0.0,
            upper_bound=1.0,
            last_updated=claim_obj.created_at,
            alpha=1.0,
            beta=1.0,
            support_evidence_by_replica={},
            refute_evidence_by_replica={},
        )
        self._recompute_summary(cs)
        self.confidence[claim_obj.id] = cs

        # Ensure dedup cache exists for the claim.
        self._seen_signal_ids_by_claim.setdefault(claim_obj.id, {})
        self._signal_events_by_claim.setdefault(claim_obj.id, {})
        self._signal_replay_dirty_by_claim.setdefault(claim_obj.id, False)
        self._signal_event_order_by_claim.setdefault(claim_obj.id, [])
        self._signal_event_replay_index_by_claim.setdefault(claim_obj.id, 0)
        self._last_source_stance_by_claim.setdefault(claim_obj.id, {})
        self._attribution_events_by_claim.setdefault(claim_obj.id, {})
        self._attribution_applied_by_claim.setdefault(claim_obj.id, {})

    def _prune_signal_events(self, claim_id: str, *, now: datetime) -> bool:
        removed = False
        if self.signal_event_max_events <= 0:
            self._signal_events_by_claim[claim_id] = {}
            return True

        events = self._signal_events_by_claim.get(claim_id)
        if events is None:
            return False

        if self.signal_event_window_seconds > 0:
            cutoff = now.timestamp() - self.signal_event_window_seconds
            for signal_id, (_sid, _pol, _str, _trust, ts) in list(events.items()):
                if ts.timestamp() < cutoff:
                    del events[signal_id]
                    removed = True

        # Enforce capacity by timestamp (deterministic even after merges).
        if self.signal_event_max_events > 0 and len(events) > self.signal_event_max_events:
            to_remove = len(events) - self.signal_event_max_events
            oldest = sorted(((payload[4], signal_id) for signal_id, payload in events.items()))[:to_remove]
            for _ts, signal_id in oldest:
                if signal_id in events:
                    del events[signal_id]
                    removed = True

        return removed

    def _recompute_claim_from_signal_events(self, claim_id: str, *, now: Optional[datetime] = None) -> None:
        if claim_id not in self.claims:
            raise KeyError(f"Unknown claim_id: {claim_id}")
        cs = self.confidence[claim_id]
        claim = self.claims[claim_id]

        events_dict = self._signal_events_by_claim.get(claim_id, {})

        def apply_event_sequence(signal_ids: List[str]) -> None:
            for sid in signal_ids:
                payload = events_dict.get(sid)
                if payload is None:
                    # Our cached order is stale (eviction/merge). Force a rebuild.
                    raise KeyError(sid)

                source_id, polarity, strength, eff_trust, ts = payload
                if ts < cs.last_updated:
                    # Out-of-order (older than our baseline). Skip rather than time-travel.
                    continue

                elapsed = (ts - cs.last_updated).total_seconds()
                if elapsed > 0:
                    cs.support_evidence_by_replica["events"] = exponential_decay(
                        float(cs.support_evidence_by_replica.get("events", 0.0)),
                        elapsed,
                        self.half_life_seconds,
                    )
                    cs.refute_evidence_by_replica["events"] = exponential_decay(
                        float(cs.refute_evidence_by_replica.get("events", 0.0)),
                        elapsed,
                        self.half_life_seconds,
                    )

                weight = max(0.0, float(strength)) * max(0.0, float(eff_trust))
                sup = max(0.0, float(polarity)) * weight
                ref = max(0.0, -float(polarity)) * weight

                cs.support_evidence_by_replica["events"] = float(cs.support_evidence_by_replica.get("events", 0.0)) + sup
                cs.refute_evidence_by_replica["events"] = float(cs.refute_evidence_by_replica.get("events", 0.0)) + ref
                cs.last_updated = ts
                self._recompute_summary(cs)

        dirty = bool(self._signal_replay_dirty_by_claim.get(claim_id, True))
        order = self._signal_event_order_by_claim.get(claim_id)
        replay_index = int(self._signal_event_replay_index_by_claim.get(claim_id, 0))
        if order is None:
            order = []
        if replay_index < 0:
            replay_index = 0

        if dirty:
            # Full rebuild: reset to prior at claim creation time.
            cs.support_evidence_by_replica = {"events": 0.0}
            cs.refute_evidence_by_replica = {"events": 0.0}
            cs.last_updated = claim.created_at
            self._recompute_summary(cs)

            order = list(events_dict.keys())
            order.sort(key=lambda sid: (events_dict[sid][4], sid))  # by timestamp then id
            apply_event_sequence(order)

            self._signal_event_order_by_claim[claim_id] = order
            self._signal_event_replay_index_by_claim[claim_id] = len(order)
            self._signal_replay_dirty_by_claim[claim_id] = False
        else:
            # Incremental: apply only events we haven't replayed yet, in cached order.
            # If caches drift (eviction/merge), fall back to a rebuild.
            try:
                if replay_index > len(order):
                    raise ValueError("replay_index out of bounds")
                if replay_index < len(order):
                    tail = order[replay_index:]
                    apply_event_sequence(tail)
                    self._signal_event_replay_index_by_claim[claim_id] = len(order)
            except (KeyError, ValueError):
                self._signal_replay_dirty_by_claim[claim_id] = True
                self._recompute_claim_from_signal_events(claim_id, now=now)
                return

        if now is not None:
            if now < cs.last_updated:
                raise ValueError("now must be >= last_updated")
            elapsed = (now - cs.last_updated).total_seconds()
            if elapsed > 0:
                cs.support_evidence_by_replica["events"] = exponential_decay(
                    float(cs.support_evidence_by_replica.get("events", 0.0)),
                    elapsed,
                    self.half_life_seconds,
                )
                cs.refute_evidence_by_replica["events"] = exponential_decay(
                    float(cs.refute_evidence_by_replica.get("events", 0.0)),
                    elapsed,
                    self.half_life_seconds,
                )
                cs.last_updated = now
                self._recompute_summary(cs)

        self.confidence[claim_id] = cs

    def _prune_attribution_events(self, claim_id: str, *, now: datetime) -> None:
        if self.outcome_attribution_max_events <= 0:
            self._attribution_events_by_claim[claim_id] = {}
            self._attribution_applied_by_claim[claim_id] = {}
            return

        events_map = self._attribution_events_by_claim.get(claim_id)
        applied_map = self._attribution_applied_by_claim.get(claim_id)
        if events_map is None or applied_map is None:
            return

        if self.outcome_attribution_window_seconds > 0:
            cutoff = now.timestamp() - self.outcome_attribution_window_seconds
            for event_id, (_rid, _sid, _pol, _str, ts) in list(events_map.items()):
                if ts.timestamp() < cutoff:
                    del events_map[event_id]
            for event_id, applied_at in list(applied_map.items()):
                if applied_at.timestamp() < cutoff:
                    del applied_map[event_id]

        # Enforce capacity by timestamp (deterministic even after merges).
        if self.outcome_attribution_max_events > 0 and len(events_map) > self.outcome_attribution_max_events:
            to_remove = len(events_map) - self.outcome_attribution_max_events
            oldest = sorted(((payload[4], event_id) for event_id, payload in events_map.items()))[:to_remove]
            for _ts, event_id in oldest:
                if event_id in events_map:
                    del events_map[event_id]

        if self.outcome_attribution_max_events > 0 and len(applied_map) > self.outcome_attribution_max_events:
            to_remove = len(applied_map) - self.outcome_attribution_max_events
            oldest = sorted(((applied_at, event_id) for event_id, applied_at in applied_map.items()))[:to_remove]
            for _ts, event_id in oldest:
                if event_id in applied_map:
                    del applied_map[event_id]

    def get_source_trust(self, source_id: str) -> float:
        return self.source_trust.get(source_id, SourceTrustState(source_id)).trust

    def update_source_trust(
        self,
        source_id: str,
        *,
        correct: bool,
        weight: float = 1.0,
        replica_id: Optional[str] = None,
    ) -> float:
        st = self.source_trust.get(source_id)
        if st is None:
            st = SourceTrustState(source_id)
            self.source_trust[source_id] = st
        st.update(correct=correct, weight=weight, replica_id=replica_id or self.replica_id)
        return st.trust
        
    def add_signal(self, signal: Union[Signal, None] = None, **kwargs):
        if signal is None:
            signal = Signal(**kwargs)

        if signal.claim_id not in self.confidence:
            raise KeyError(f"Unknown claim_id: {signal.claim_id}")

        if self._should_dedup(signal):
            claim_cache = self._seen_signal_ids_by_claim.setdefault(signal.claim_id, {})
            self._prune_dedup_cache(claim_cache, now=signal.timestamp)

            prev = claim_cache.get(signal.signal_id)  # type: ignore[arg-type]
            if prev is not None:
                # Duplicate within our memory window -> ignore.
                return

            # Record the id before applying updates (so retries during processing are also deduped).
            claim_cache[signal.signal_id] = signal.timestamp  # type: ignore[index]
            self._enforce_dedup_capacity(claim_cache)

        # Determine a globally unique signal id.
        signal_id = getattr(signal, "signal_id", None)
        if not signal_id:
            self._signal_event_counter += 1
            signal_id = f"{self.replica_id}:{self._signal_event_counter}"
            signal.signal_id = signal_id

        # Determine effective trust to store with the event (deterministic replay across replicas).
        eff_trust = signal.source_trust
        if eff_trust is None:
            eff_trust = self.get_source_trust(signal.source_id)
        eff_trust = float(eff_trust)

        events = self._signal_events_by_claim.setdefault(signal.claim_id, {})
        # If this event's timestamp is <= current computed last_updated, it is out-of-order
        # and requires a full rebuild to remain correct.
        current_now = self.confidence[signal.claim_id].last_updated
        if signal.timestamp <= current_now:
            self._signal_replay_dirty_by_claim[signal.claim_id] = True
        existing = events.get(signal_id)
        if existing is None:
            events[signal_id] = (
                signal.source_id,
                float(signal.polarity),
                float(signal.strength),
                eff_trust,
                signal.timestamp,
            )
            # Fast-path: update cached order for in-order appends.
            order = self._signal_event_order_by_claim.setdefault(signal.claim_id, [])
            dirty = bool(self._signal_replay_dirty_by_claim.get(signal.claim_id, True))
            if not dirty and order:
                last_id = order[-1]
                last_payload = events.get(last_id)
                if last_payload is None:
                    self._signal_replay_dirty_by_claim[signal.claim_id] = True
                else:
                    last_ts = last_payload[4]
                    if (signal.timestamp, signal_id) > (last_ts, last_id):
                        order.append(signal_id)
                    else:
                        self._signal_replay_dirty_by_claim[signal.claim_id] = True
            elif not dirty and not order:
                # No cache yet; force rebuild to initialize it.
                self._signal_replay_dirty_by_claim[signal.claim_id] = True

            pruned = self._prune_signal_events(signal.claim_id, now=signal.timestamp)
            if pruned:
                # Eviction changes the replay set; rebuild to keep derived confidence correct.
                self._signal_replay_dirty_by_claim[signal.claim_id] = True
                self._signal_event_order_by_claim[signal.claim_id] = []
                self._signal_event_replay_index_by_claim[signal.claim_id] = 0
        else:
            # If a duplicate arrives, it must match exactly.
            if existing != (
                signal.source_id,
                float(signal.polarity),
                float(signal.strength),
                eff_trust,
                signal.timestamp,
            ):
                raise ValueError(f"Conflicting signal payload for signal_id={signal_id}")

        # Recompute confidence deterministically from the event log.
        # If the signal is out-of-order, recompute to the existing "now" to avoid time-travel.
        recompute_now = signal.timestamp if signal.timestamp >= current_now else current_now
        self._recompute_claim_from_signal_events(signal.claim_id, now=recompute_now)

        polarity = float(signal.polarity)
        weight = max(0.0, float(signal.strength)) * max(0.0, float(eff_trust))

        # Store last stance for outcome feedback.
        if polarity != 0 and weight > 0:
            self._last_source_stance_by_claim.setdefault(signal.claim_id, {})[signal.source_id] = (
                polarity,
                max(0.0, float(signal.strength)),
                signal.timestamp,
            )

            # Record mergeable attribution event (bounded).
            event_id = getattr(signal, "signal_id", None)
            if not event_id:
                self._attribution_event_counter += 1
                event_id = f"{self.replica_id}:{self._attribution_event_counter}"

            applied = self._attribution_applied_by_claim.setdefault(signal.claim_id, {})
            if event_id not in applied:
                events = self._attribution_events_by_claim.setdefault(signal.claim_id, {})
                if event_id not in events:
                    events[event_id] = (
                        self.replica_id,
                        signal.source_id,
                        polarity,
                        max(0.0, float(signal.strength)),
                        signal.timestamp,
                    )
                    self._prune_attribution_events(signal.claim_id, now=signal.timestamp)

    def apply_outcome(
        self,
        claim_id: str,
        *,
        outcome: bool,
        now: Optional[datetime] = None,
        weight: float = 1.0,
        clear_attribution: bool = True,
    ):
        """Apply ground-truth outcome feedback to update learned source trust.

        This uses the most recent stance per source for the claim (bounded memory).
        """
        if claim_id not in self.claims:
            raise KeyError(f"Unknown claim_id: {claim_id}")

        t = now or datetime.now()

        events_map = self._attribution_events_by_claim.get(claim_id, {})
        applied_map = self._attribution_applied_by_claim.get(claim_id, {})
        self._prune_attribution_events(claim_id, now=t)

        if events_map:
            # Update trust based on all recent events. Credit updates to the event's origin replica
            # so distributed merges remain idempotent (via per-replica max).
            for event_id, (origin_replica, source_id, polarity, strength, ts) in list(events_map.items()):
                if event_id in applied_map:
                    continue
                if ts > t:
                    continue
                if polarity == 0:
                    continue

                predicted_true = polarity > 0
                correct = (predicted_true is True and outcome is True) or (predicted_true is False and outcome is False)
                self.update_source_trust(
                    source_id,
                    correct=correct,
                    weight=float(weight) * max(0.0, float(strength)),
                    replica_id=origin_replica,
                )

                if clear_attribution:
                    applied_map[event_id] = t
                    del events_map[event_id]

            if clear_attribution:
                # Also clear last-stance fallback to avoid double-application.
                self._last_source_stance_by_claim[claim_id] = {}

            self._attribution_events_by_claim[claim_id] = events_map
            self._attribution_applied_by_claim[claim_id] = applied_map
            self._prune_attribution_events(claim_id, now=t)
        else:
            # If we have tombstones for this claim, do not fall back to last-stance updates;
            # that would re-apply outcomes for already-processed events after merges.
            if applied_map:
                return
            # Fallback: bounded last-stance memory.
            stances = self._last_source_stance_by_claim.get(claim_id, {})
            for source_id, (polarity, strength, _ts) in stances.items():
                if polarity == 0:
                    continue
                predicted_true = polarity > 0
                correct = (predicted_true is True and outcome is True) or (predicted_true is False and outcome is False)
                self.update_source_trust(
                    source_id,
                    correct=correct,
                    weight=float(weight) * max(0.0, float(strength)),
                    replica_id=self.replica_id,
                )

    def _should_dedup(self, signal: Signal) -> bool:
        if self.dedup_window_seconds <= 0:
            return False
        if self.dedup_max_per_claim <= 0:
            return False
        return bool(getattr(signal, "signal_id", None))

    def _prune_dedup_cache(self, cache: Dict[str, datetime], *, now: datetime) -> None:
        if self.dedup_window_seconds <= 0:
            return
        cutoff = now.timestamp() - self.dedup_window_seconds
        # dict preserves insertion order; prune from oldest to newest.
        for sid in list(cache.keys()):
            if cache[sid].timestamp() < cutoff:
                del cache[sid]
            else:
                break

    def _enforce_dedup_capacity(self, cache: Dict[str, datetime]) -> None:
        if self.dedup_max_per_claim <= 0:
            cache.clear()
            return
        while len(cache) > self.dedup_max_per_claim:
            # Evict oldest inserted entry.
            oldest_key = next(iter(cache.keys()))
            del cache[oldest_key]

    def tick(self, now: datetime):
        for claim_id in list(self.confidence.keys()):
            self._recompute_claim_from_signal_events(claim_id, now=now)

    def export_snapshot(self, now: Optional[datetime] = None):
        """Export a JSON-serializable snapshot of confidence states.

        If now is provided, all states are decayed to that time before export.
        """
        if now is not None:
            self.tick(now)

        return {
            "created_at": (now or datetime.now()).isoformat(),
            "half_life_seconds": float(self.half_life_seconds),
            "claims": {claim_id: snapshot_from_state(cs) for claim_id, cs in self.confidence.items()},
            "signal_events": {
                claim_id: {
                    signal_id: {
                        "source_id": source_id,
                        "polarity": polarity,
                        "strength": strength,
                        "effective_trust": eff_trust,
                        "timestamp": ts.isoformat(),
                    }
                    for signal_id, (source_id, polarity, strength, eff_trust, ts) in events.items()
                }
                for claim_id, events in self._signal_events_by_claim.items()
            },
            "trust": {source_id: snapshot_from_trust_state(st) for source_id, st in self.source_trust.items()},
            "attribution": {
                claim_id: {
                    event_id: {
                        "origin_replica_id": rid,
                        "source_id": sid,
                        "polarity": pol,
                        "strength": strength,
                        "timestamp": ts.isoformat(),
                    }
                    for event_id, (rid, sid, pol, strength, ts) in events.items()
                }
                for claim_id, events in self._attribution_events_by_claim.items()
            },
            "attribution_applied": {
                claim_id: {event_id: applied_at.isoformat() for event_id, applied_at in applied.items()}
                for claim_id, applied in self._attribution_applied_by_claim.items()
            },
        }

    def merge_snapshot(
        self,
        snapshot: dict,
        *,
        now: Optional[datetime] = None,
        strategy: MergeStrategy = "conservative_max",
    ):
        """Merge a remote snapshot into this engine.

        Strategies:
        - conservative_max: avoids double-counting (safe default), but can under-aggregate
        - additive_sum: assumes disjoint evidence across nodes; can overcount if snapshots overlap
        """
        merge_time = now or datetime.now()

        remote_half_life = float(snapshot.get("half_life_seconds", self.half_life_seconds))
        if abs(remote_half_life - float(self.half_life_seconds)) > 1e-9:
            # Keep local half-life as the canonical one; decay remote evidence using local half-life.
            pass

        # Merge signal events (authoritative) if present.
        remote_signal_events = snapshot.get("signal_events")
        if remote_signal_events is not None:
            for claim_id, events_map in remote_signal_events.items():
                if claim_id not in self.claims:
                    self.define_claim(claim_id, description="(imported)", created_at=merge_time)

                local_events = self._signal_events_by_claim.setdefault(claim_id, {})
                inserted_any = False
                for signal_id, ev in events_map.items():
                    payload = (
                        ev["source_id"],
                        float(ev["polarity"]),
                        float(ev["strength"]),
                        float(ev["effective_trust"]),
                        datetime.fromisoformat(ev["timestamp"]),
                    )
                    existing = local_events.get(signal_id)
                    if existing is None:
                        local_events[signal_id] = payload
                        inserted_any = True
                        if payload[4] <= self.confidence[claim_id].last_updated:
                            self._signal_replay_dirty_by_claim[claim_id] = True
                    elif existing != payload:
                        raise ValueError(f"Conflicting signal payload for signal_id={signal_id}")

                pruned = self._prune_signal_events(claim_id, now=merge_time)
                if inserted_any or pruned:
                    # Union merges can introduce out-of-order timestamps relative to local cache.
                    self._signal_replay_dirty_by_claim[claim_id] = True
                    self._signal_event_order_by_claim[claim_id] = []
                    self._signal_event_replay_index_by_claim[claim_id] = 0
                self._recompute_claim_from_signal_events(claim_id, now=merge_time)
        else:
            # Backward-compat: merge confidence snapshots directly.
            remote_claims = snapshot.get("claims", {})
            for claim_id, remote_cs in remote_claims.items():
                if claim_id not in self.claims:
                    self.define_claim(claim_id, description="(imported)", created_at=merge_time)

                local_snapshot = snapshot_from_state(self.confidence[claim_id])
                merged = merge_confidence_snapshots(
                    local=local_snapshot,
                    remote=remote_cs,
                    now=merge_time,
                    half_life_seconds=float(self.half_life_seconds),
                    strategy=strategy,
                )

                cs = self.confidence[claim_id]
                cs.support_evidence_by_replica = {
                    k: float(v) for k, v in merged.get("support_evidence_by_replica", {}).items()
                }
                cs.refute_evidence_by_replica = {
                    k: float(v) for k, v in merged.get("refute_evidence_by_replica", {}).items()
                }
                cs.last_updated = merge_time
                self._recompute_summary(cs)
                self.confidence[claim_id] = cs

        # Merge learned trust (optional).
        remote_trust = snapshot.get("trust", {})
        for source_id, remote_st in remote_trust.items():
            local_st = self.source_trust.get(source_id)
            if local_st is None:
                local_st = SourceTrustState(source_id)
                self.source_trust[source_id] = local_st

            local_snap = snapshot_from_trust_state(local_st)
            merged = merge_trust_snapshots(local=local_snap, remote=remote_st, strategy=strategy)

            local_st.correct_by_replica = {k: float(v) for k, v in merged.get("correct_by_replica", {}).items()}
            local_st.incorrect_by_replica = {k: float(v) for k, v in merged.get("incorrect_by_replica", {}).items()}
            local_st.recompute()

        # Merge attribution OR-set (events + applied tombstones).
        merge_time = now or datetime.now()
        remote_applied = snapshot.get("attribution_applied", {})
        remote_events = snapshot.get("attribution", {})

        # First merge applied tombstones (max applied_at).
        for claim_id, applied_map in remote_applied.items():
            self._attribution_applied_by_claim.setdefault(claim_id, {})
            local_applied = self._attribution_applied_by_claim[claim_id]
            for event_id, applied_at_iso in applied_map.items():
                applied_at = datetime.fromisoformat(applied_at_iso)
                prev = local_applied.get(event_id)
                if prev is None or applied_at > prev:
                    local_applied[event_id] = applied_at
            if applied_map:
                # Prevent last-stance fallback from re-applying already-tombstoned events.
                self._last_source_stance_by_claim[claim_id] = {}

        # Then merge live events; drop any that are applied.
        for claim_id, events_map in remote_events.items():
            self._attribution_events_by_claim.setdefault(claim_id, {})
            self._attribution_applied_by_claim.setdefault(claim_id, {})
            local_events = self._attribution_events_by_claim[claim_id]
            local_applied = self._attribution_applied_by_claim[claim_id]

            for event_id, ev in events_map.items():
                if event_id in local_applied:
                    continue
                origin = ev.get("origin_replica_id", "legacy")
                source_id = ev["source_id"]
                polarity = float(ev["polarity"])
                strength = float(ev["strength"])
                ts = datetime.fromisoformat(ev["timestamp"])

                if event_id not in local_events:
                    local_events[event_id] = (origin, source_id, polarity, strength, ts)

            # Enforce bounds
            self._prune_attribution_events(claim_id, now=merge_time)

        # Finally, ensure any locally-stored events that are now tombstoned are removed.
        for claim_id, local_applied in self._attribution_applied_by_claim.items():
            local_events = self._attribution_events_by_claim.setdefault(claim_id, {})
            for event_id in list(local_events.keys()):
                if event_id in local_applied:
                    del local_events[event_id]
        
    def get_confidence(self, claim_id: str, now: Optional[datetime] = None) -> float:
        if claim_id not in self.confidence:
            raise KeyError(f"Unknown claim_id: {claim_id}")
        if now is not None:
            self._recompute_claim_from_signal_events(claim_id, now=now)
        return self.confidence[claim_id].value

    def get_confidence_interval(
        self,
        claim_id: str,
        now: Optional[datetime] = None,
    ):
        if claim_id not in self.confidence:
            raise KeyError(f"Unknown claim_id: {claim_id}")
        if now is not None:
            self._recompute_claim_from_signal_events(claim_id, now=now)
        cs = self.confidence[claim_id]
        return cs.lower_bound, cs.upper_bound

    def get_confidence_state(self, claim_id: str, now: Optional[datetime] = None) -> ConfidenceState:
        if claim_id not in self.confidence:
            raise KeyError(f"Unknown claim_id: {claim_id}")
        if now is not None:
            self._recompute_claim_from_signal_events(claim_id, now=now)
        return self.confidence[claim_id]