"""Retrospective identity memory built from immutable ReID evidence.

The local detections and model responses are observations.  Identity clusters
are a projection of those observations and may be rebuilt when later, stronger
evidence contradicts an earlier merge.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable


class ConstrainedUnionFind:
    """Union-find used for one disposable identity-memory projection."""

    def __init__(self, cannot_link: set[tuple[int, int]] | None = None) -> None:
        self.parent: dict[str, str] = {}
        self.astra: dict[str, dict[int, int]] = {}
        self.local_ids: dict[str, set[int]] = {}
        self.cannot_link = cannot_link or set()
        self.last_rejection: str | None = None

    def add(
        self,
        node: str,
        batch: int | None = None,
        aid: int | None = None,
        local_id: int | None = None,
    ) -> None:
        if node not in self.parent:
            self.parent[node] = node
            self.astra[node] = (
                {batch: aid} if batch is not None and aid is not None else {}
            )
            self.local_ids[node] = {local_id} if local_id is not None else set()

    def find(self, node: str) -> str:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, left: str, right: str) -> bool:
        self.last_rejection = None
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return True
        left_astra, right_astra = self.astra[left_root], self.astra[right_root]
        for batch, identity_id in right_astra.items():
            if batch in left_astra and left_astra[batch] != identity_id:
                self.last_rejection = "astra_conflict"
                return False
        if any(
            tuple(sorted((left_id, right_id))) in self.cannot_link
            for left_id in self.local_ids[left_root]
            for right_id in self.local_ids[right_root]
        ):
            self.last_rejection = "temporal_overlap"
            return False
        self.parent[right_root] = left_root
        left_astra.update(right_astra)
        self.local_ids[left_root].update(self.local_ids[right_root])
        del self.astra[right_root]
        del self.local_ids[right_root]
        return True


@dataclass(frozen=True)
class IdentityEvidence:
    """One candidate association retained independently of its current verdict."""

    tracklet_id: int
    batch: int
    identity_id: int
    source: str
    support: int
    confidence: float
    weighted_support: float
    role: str = "unknown"
    required_support: int | None = None
    required_confidence: float | None = None
    policy: str | None = None
    reference_tracklet_id: int | None = None
    confirming_batch: int | None = None

    @property
    def key(self) -> tuple[int, int, int]:
        return self.tracklet_id, self.batch, self.identity_id

    def decision_base(self) -> dict[str, Any]:
        decision = {
            "tracklet_id": self.tracklet_id,
            "batch": self.batch,
            "identity_id": self.identity_id,
            "source": self.source,
            "support": self.support,
            "confidence": round(self.confidence, 4),
            "weighted_support": round(self.weighted_support, 4),
        }
        if self.policy is not None:
            decision.update(
                {
                    "policy": self.policy,
                    "required_support": self.required_support,
                    "required_confidence": round(
                        float(self.required_confidence), 4
                    ),
                    "reference_tracklet_id": self.reference_tracklet_id,
                    "confirming_batch": self.confirming_batch,
                }
            )
        return decision


@dataclass
class IdentityProjection:
    union_find: ConstrainedUnionFind
    decisions: list[dict[str, Any]]
    rejected_reasons: Counter[str]


class RetrospectiveIdentityMemory:
    """Accumulate evidence and rebuild identity hypotheses after every batch.

    Eligible evidence is replayed strongest-first.  Consequently, a strong
    later observation can make an older weak merge invalid without mutating the
    underlying observations or requiring a rollback-capable union-find.
    """

    def __init__(
        self,
        local_ids: Iterable[int],
        *,
        cannot_link: set[tuple[int, int]] | None = None,
        merge_confidence: float,
        min_support: int,
    ) -> None:
        self.local_ids = set(local_ids)
        self.cannot_link = cannot_link or set()
        self.merge_confidence = merge_confidence
        self.min_support = min_support
        self._evidence: list[IdentityEvidence] = []
        self._last_decisions: dict[tuple[int, int, int], dict[str, Any]] = {}
        self.revision_log: list[dict[str, Any]] = []

    def observe_batch(self, evidence: Iterable[IdentityEvidence], *, batch: int) -> None:
        additions = list(evidence)
        if any(item.batch != batch for item in additions):
            raise ValueError("all identity evidence must belong to the observed batch")
        self._evidence.extend(additions)
        projection = self.resolve()
        current = {
            (
                int(decision["tracklet_id"]),
                int(decision["batch"]),
                int(decision["identity_id"]),
            ): decision
            for decision in projection.decisions
        }
        for key, previous in self._last_decisions.items():
            updated = current.get(key)
            if updated is None or previous["decision"] == updated["decision"]:
                continue
            self.revision_log.append(
                {
                    "trigger_batch": batch,
                    "tracklet_id": key[0],
                    "evidence_batch": key[1],
                    "identity_id": key[2],
                    "from": previous["decision"],
                    "to": updated["decision"],
                    "confidence": updated["confidence"],
                    "trigger_confidence": round(
                        max(item.confidence for item in additions), 4
                    ),
                    **(
                        {"reason": updated["reason"]}
                        if "reason" in updated
                        else {}
                    ),
                }
            )
        self._last_decisions = current

    def resolve(self) -> IdentityProjection:
        union_find = ConstrainedUnionFind(self.cannot_link)
        for tracklet_id in self.local_ids:
            union_find.add(f"T{tracklet_id}", local_id=tracklet_id)

        decisions: dict[tuple[int, int, int], dict[str, Any]] = {}
        eligible: list[IdentityEvidence] = []
        for item in self._evidence:
            decision = item.decision_base()
            required_support = (
                item.required_support
                if item.required_support is not None
                else self.min_support
            )
            required_confidence = (
                item.required_confidence
                if item.required_confidence is not None
                else self.merge_confidence
            )
            if (
                item.support < required_support
                or item.confidence < required_confidence
                or item.weighted_support < required_support * required_confidence
            ):
                decision.update(
                    {"decision": "abstain", "reason": "insufficient_evidence"}
                )
                decisions[item.key] = decision
            else:
                eligible.append(item)

        # Replay by strength, not arrival time.  Stable trailing keys keep the
        # output deterministic when two pieces of evidence are equally strong.
        eligible.sort(
            key=lambda item: (
                -item.confidence,
                -item.weighted_support,
                -item.support,
                item.batch,
                item.tracklet_id,
                item.identity_id,
            )
        )
        rejected_reasons: Counter[str] = Counter()
        for item in eligible:
            identity_node = f"A{item.batch}:{item.identity_id}"
            union_find.add(identity_node, item.batch, item.identity_id)
            decision = item.decision_base()
            if union_find.union(f"T{item.tracklet_id}", identity_node):
                decision["decision"] = "merge"
            else:
                reason = union_find.last_rejection or "unknown"
                decision.update({"decision": "reject", "reason": reason})
                rejected_reasons[reason] += 1
            decisions[item.key] = decision

        return IdentityProjection(
            union_find=union_find,
            decisions=sorted(
                decisions.values(),
                key=lambda item: (item["batch"], item["tracklet_id"]),
            ),
            rejected_reasons=rejected_reasons,
        )
