from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Literal, TypedDict

from ..confidence.confidence_state import ConfidenceState
from ..decay.decay_function import exponential_decay


MergeStrategy = Literal["conservative_max", "additive_sum"]


class ConfidenceStateSnapshot(TypedDict):
    claim_id: str
    # Excess evidence beyond the prior Beta(1,1), tracked per replica.
    support_evidence_by_replica: Dict[str, float]
    refute_evidence_by_replica: Dict[str, float]
    # Backward-compat: older snapshots may include only alpha/beta.
    alpha: float
    beta: float
    last_updated: str  # ISO-8601


@dataclass(frozen=True)
class EngineSnapshot:
    created_at: datetime
    half_life_seconds: float
    claims: Dict[str, ConfidenceStateSnapshot]


def _decay_map_to_now(evidence_by_replica: Dict[str, float], last_updated: datetime, now: datetime, half_life_seconds: float):
    elapsed = (now - last_updated).total_seconds()
    if elapsed < 0:
        raise ValueError("Snapshot time is in the future relative to now")

    out: Dict[str, float] = {}
    for rid, value in evidence_by_replica.items():
        out[rid] = exponential_decay(max(0.0, float(value)), elapsed, half_life_seconds)
    return out


def snapshot_from_state(cs: ConfidenceState) -> ConfidenceStateSnapshot:
    return {
        "claim_id": cs.claim_id,
        "support_evidence_by_replica": {k: float(v) for k, v in cs.support_evidence_by_replica.items()},
        "refute_evidence_by_replica": {k: float(v) for k, v in cs.refute_evidence_by_replica.items()},
        "alpha": float(cs.alpha),
        "beta": float(cs.beta),
        "last_updated": cs.last_updated.isoformat(),
    }


def _normalize_snapshot(snapshot: ConfidenceStateSnapshot) -> ConfidenceStateSnapshot:
    # Ensure per-replica maps exist; fall back to a single legacy bucket if only alpha/beta exist.
    if "support_evidence_by_replica" in snapshot and "refute_evidence_by_replica" in snapshot:
        return snapshot

    alpha = float(snapshot.get("alpha", 1.0))
    beta = float(snapshot.get("beta", 1.0))
    return {
        "claim_id": snapshot["claim_id"],
        "support_evidence_by_replica": {"legacy": max(0.0, alpha - 1.0)},
        "refute_evidence_by_replica": {"legacy": max(0.0, beta - 1.0)},
        "alpha": alpha,
        "beta": beta,
        "last_updated": snapshot["last_updated"],
    }


def merge_confidence_snapshots(
    *,
    local: ConfidenceStateSnapshot,
    remote: ConfidenceStateSnapshot,
    now: datetime,
    half_life_seconds: float,
    strategy: MergeStrategy,
) -> ConfidenceStateSnapshot:
    local = _normalize_snapshot(local)
    remote = _normalize_snapshot(remote)

    if local["claim_id"] != remote["claim_id"]:
        raise ValueError("Cannot merge snapshots for different claim_ids")

    local_t = datetime.fromisoformat(local["last_updated"])
    remote_t = datetime.fromisoformat(remote["last_updated"])

    l_sup = _decay_map_to_now(local["support_evidence_by_replica"], local_t, now, half_life_seconds)
    l_ref = _decay_map_to_now(local["refute_evidence_by_replica"], local_t, now, half_life_seconds)
    r_sup = _decay_map_to_now(remote["support_evidence_by_replica"], remote_t, now, half_life_seconds)
    r_ref = _decay_map_to_now(remote["refute_evidence_by_replica"], remote_t, now, half_life_seconds)

    merged_sup: Dict[str, float] = {}
    merged_ref: Dict[str, float] = {}

    if strategy == "conservative_max":
        for rid in set(l_sup.keys()) | set(r_sup.keys()):
            merged_sup[rid] = max(float(l_sup.get(rid, 0.0)), float(r_sup.get(rid, 0.0)))
        for rid in set(l_ref.keys()) | set(r_ref.keys()):
            merged_ref[rid] = max(float(l_ref.get(rid, 0.0)), float(r_ref.get(rid, 0.0)))
    elif strategy == "additive_sum":
        for rid in set(l_sup.keys()) | set(r_sup.keys()):
            merged_sup[rid] = float(l_sup.get(rid, 0.0)) + float(r_sup.get(rid, 0.0))
        for rid in set(l_ref.keys()) | set(r_ref.keys()):
            merged_ref[rid] = float(l_ref.get(rid, 0.0)) + float(r_ref.get(rid, 0.0))
    else:
        raise ValueError(f"Unknown merge strategy: {strategy}")

    alpha = 1.0 + sum(merged_sup.values())
    beta = 1.0 + sum(merged_ref.values())

    return {
        "claim_id": local["claim_id"],
        "support_evidence_by_replica": merged_sup,
        "refute_evidence_by_replica": merged_ref,
        "alpha": float(alpha),
        "beta": float(beta),
        "last_updated": now.isoformat(),
    }
