"""Offline identity graph assembled from complementary ReID evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from flowmetric.identity.appearance import AppearanceStitch
from flowmetric.identity.stitch import TrackletStitch


@dataclass(frozen=True)
class IdentityGraphEdge:
    left_id: int
    right_id: int
    confidence: float
    sources: tuple[str, ...]
    appearance_similarity: float | None = None
    motion_score: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "left_tracklet_id": self.left_id,
            "right_tracklet_id": self.right_id,
            "confidence": round(self.confidence, 4),
            "sources": list(self.sources),
            "appearance_similarity": (
                round(self.appearance_similarity, 4)
                if self.appearance_similarity is not None
                else None
            ),
            "motion_score": (
                round(self.motion_score, 4)
                if self.motion_score is not None
                else None
            ),
        }


def build_identity_graph_edges(
    mutual_appearance: Iterable[AppearanceStitch],
    supplemental_appearance: Iterable[AppearanceStitch],
    motion: Iterable[TrackletStitch],
    scenario_appearance: Iterable[AppearanceStitch] = (),
) -> list[IdentityGraphEdge]:
    """Deduplicate pair evidence and rank strongest multi-source edges first."""
    evidence: dict[tuple[int, int], dict[str, object]] = {}

    def item(left: int, right: int) -> dict[str, object]:
        pair = tuple(sorted((int(left), int(right))))
        return evidence.setdefault(
            pair,
            {
                "left_id": pair[0],
                "right_id": pair[1],
                "sources": set(),
                "appearance_similarity": None,
                "motion_score": None,
            },
        )

    for proposal in mutual_appearance:
        current = item(proposal.earlier_id, proposal.later_id)
        current["sources"].add("appearance_mutual")
        current["appearance_similarity"] = max(
            float(current["appearance_similarity"] or -1.0),
            float(proposal.similarity),
        )
    for proposal in supplemental_appearance:
        current = item(proposal.earlier_id, proposal.later_id)
        current["sources"].add("appearance_strict")
        current["appearance_similarity"] = max(
            float(current["appearance_similarity"] or -1.0),
            float(proposal.similarity),
        )
    for proposal in scenario_appearance:
        current = item(proposal.earlier_id, proposal.later_id)
        current["sources"].add("appearance_scenario")
        current["appearance_similarity"] = max(
            float(current["appearance_similarity"] or -1.0),
            float(proposal.similarity),
        )
    for proposal in motion:
        current = item(proposal.earlier_id, proposal.later_id)
        current["sources"].add("motion")
        prior = current["motion_score"]
        current["motion_score"] = (
            float(proposal.score)
            if prior is None
            else min(float(prior), float(proposal.score))
        )

    edges: list[IdentityGraphEdge] = []
    for current in evidence.values():
        appearance = current["appearance_similarity"]
        motion_score = current["motion_score"]
        scores = []
        if appearance is not None:
            scores.append(float(appearance))
        if motion_score is not None:
            scores.append(max(0.0, 1.0 - float(motion_score)))
        sources = tuple(sorted(current["sources"]))
        confidence = max(scores) + (0.02 if len(sources) > 1 else 0.0)
        edges.append(
            IdentityGraphEdge(
                left_id=int(current["left_id"]),
                right_id=int(current["right_id"]),
                confidence=min(confidence, 1.0),
                sources=sources,
                appearance_similarity=(
                    float(appearance) if appearance is not None else None
                ),
                motion_score=(
                    float(motion_score) if motion_score is not None else None
                ),
            )
        )
    return sorted(
        edges,
        key=lambda edge: (
            -len(edge.sources),
            -edge.confidence,
            edge.left_id,
            edge.right_id,
        ),
    )
