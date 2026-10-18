from __future__ import annotations

import unittest

from flowmetric.identity.memory import IdentityEvidence, RetrospectiveIdentityMemory


def evidence(
    tracklet_id: int,
    batch: int,
    identity_id: int,
    confidence: float,
) -> IdentityEvidence:
    return IdentityEvidence(
        tracklet_id=tracklet_id,
        batch=batch,
        identity_id=identity_id,
        source="test",
        support=1,
        confidence=confidence,
        weighted_support=confidence,
    )


class RetrospectiveIdentityMemoryTests(unittest.TestCase):
    def test_later_strong_evidence_revises_earlier_weak_merge(self) -> None:
        memory = RetrospectiveIdentityMemory(
            [1, 2], merge_confidence=0.55, min_support=1
        )
        memory.observe_batch(
            [evidence(1, 0, 10, 0.60), evidence(2, 0, 10, 0.60)],
            batch=0,
        )
        initial = memory.resolve()
        self.assertEqual(
            initial.union_find.find("T1"), initial.union_find.find("T2")
        )

        memory.observe_batch(
            [evidence(1, 1, 20, 0.95), evidence(2, 1, 21, 0.95)],
            batch=1,
        )
        revised = memory.resolve()

        self.assertNotEqual(
            revised.union_find.find("T1"), revised.union_find.find("T2")
        )
        self.assertEqual(len(memory.revision_log), 1)
        self.assertEqual(memory.revision_log[0]["from"], "merge")
        self.assertEqual(memory.revision_log[0]["to"], "reject")
        self.assertEqual(memory.revision_log[0]["trigger_batch"], 1)
        self.assertEqual(memory.revision_log[0]["trigger_confidence"], 0.95)
        self.assertEqual(memory.revision_log[0]["reason"], "astra_conflict")

    def test_insufficient_evidence_remains_an_abstention(self) -> None:
        memory = RetrospectiveIdentityMemory(
            [1], merge_confidence=0.70, min_support=2
        )
        memory.observe_batch([evidence(1, 0, 10, 0.90)], batch=0)

        projection = memory.resolve()

        self.assertEqual(projection.decisions[0]["decision"], "abstain")
        self.assertEqual(
            projection.decisions[0]["reason"], "insufficient_evidence"
        )
        self.assertEqual(memory.revision_log, [])


if __name__ == "__main__":
    unittest.main()
