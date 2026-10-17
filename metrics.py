#!/usr/bin/env python3
"""Score FlowMetric output against benchmark/ground_truth.json (Timings.txt).

GT persons are tied to predicted IDs through head labels on two reference frames
(First.png = frame 0, Second.png = frame 86). Metrics:
  - anchoring:  GT people found by the detector on the reference frames
  - identity:   predicted ID count and end-to-end ID continuity
  - roles:      end-to-end staff/customer accuracy and staff detection counts
  - events:     P/R/F1 + timing error; strict = type + people + time, loose = type + time
  - states:     temporal IoU for waiting / consuming
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
GT_PATH = ROOT / "benchmark" / "ground_truth.json"
OUT = ROOT / "data" / "output"

SHORT_TRACK_PENALTY = 40.0  # px; prefer persistent tracks over sub-second fragments at anchors


def map_gt_to_pred(gt: dict[str, Any], tracks: dict[str, Any]) -> dict[int, str]:
    fps = tracks["fps"]
    by_frame: dict[int, list[tuple[str, list[float], float]]] = {}
    for gid, t in tracks["tracks"].items():
        life = (t["frames"][-1] - t["frames"][0]) / fps
        for f, b in zip(t["frames"], t["boxes"]):
            by_frame.setdefault(f, []).append((gid, b, life))

    def nested_fragment(c: tuple[str, list[float], float], others: list[tuple[str, list[float], float]]) -> bool:
        x1, y1, x2, y2 = c[1]
        area = max(1.0, (x2 - x1) * (y2 - y1))
        for o in others:
            if o is c or o[2] < 1.0:
                continue
            ox1, oy1, ox2, oy2 = o[1]
            inter = max(0.0, min(x2, ox2) - max(x1, ox1)) * max(0.0, min(y2, oy2) - max(y1, oy1))
            if inter / area > 0.8:
                return True
        return False

    mapping: dict[int, str] = {}
    used: set[str] = set()
    for fi, pts in gt["anchors"].items():
        if not fi.isdigit():
            continue
        frame_c = by_frame.get(int(fi), [])
        cands = [
            c for c in frame_c
            if c[0] not in used and not (c[2] < 1.0 and nested_fragment(c, frame_c))
        ]
        people = [(int(p), xy) for p, xy in pts.items()]
        if not cands:
            continue
        cost = np.full((len(people), len(cands)), 1e6)
        for i, (_, (x, y)) in enumerate(people):
            for j, (_, (x1, y1, x2, y2), life) in enumerate(cands):
                if x1 - 10 <= x <= x2 + 10 and y1 - 30 <= y <= y1 + 0.5 * (y2 - y1):
                    d = np.hypot(x - (x1 + x2) / 2, y - (y1 + 0.12 * (y2 - y1)))
                    cost[i, j] = d + (SHORT_TRACK_PENALTY if life < 1.0 else 0.0)
        for i, j in zip(*linear_sum_assignment(cost)):
            if cost[i, j] < 1e6:
                mapping[people[i][0]] = cands[j][0]
                used.add(cands[j][0])
    return mapping


def gt_person_ids(gt: dict[str, Any]) -> set[int]:
    """Return every person explicitly referenced by the ground truth.

    Anchor labels are the primary source, but collecting references from roles,
    events and states keeps the scorer correct for sparse/non-contiguous labels
    and for ground truth where a person has no usable anchor.
    """
    people: set[int] = set()
    for frame, points in gt.get("anchors", {}).items():
        if str(frame).isdigit():
            people.update(int(person) for person in points)
    people.update(int(person) for person in gt.get("staff", []))
    for event in gt.get("events", []):
        for field in ("person", "staff", "customer"):
            if event.get(field) is not None:
                people.add(int(event[field]))
    for state in gt.get("states", []):
        persons = state.get("persons", [state.get("person")])
        people.update(int(person) for person in persons if person is not None)
    return people


def gt_presence(gt: dict[str, Any]) -> dict[int, tuple[float, float]]:
    """Expected on-screen interval per GT person (whole clip unless an event says otherwise)."""
    dur = gt["duration_s"]
    pres = {person: [0.0, dur] for person in gt_person_ids(gt)}
    for e in gt.get("events", []):
        if e["type"] == "exit":
            pres[int(e["person"])][1] = e["t"]
        if e["type"] == "leave_kitchen":
            pres[int(e["person"])][0] = e["t"]
    return {p: (a, b) for p, (a, b) in pres.items()}


def score_events(
    gt_events: list[dict[str, Any]],
    pred_events: list[dict[str, Any]],
    pred2gt: dict[str, int],
    *,
    tol: float,
    strict: bool,
) -> dict[str, Any]:
    def people(e: dict[str, Any], to_gt: bool) -> tuple:
        keys = ("staff", "customer") if "staff" in e else ("person",)
        if to_gt:
            return tuple(pred2gt.get(str(e[k])) for k in keys)
        return tuple(e[k] for k in keys)

    types = sorted({e["type"] for e in gt_events} | {e["type"] for e in pred_events})
    per_type: dict[str, Any] = {}
    tot_tp = tot_fp = tot_fn = 0
    errs_all: list[float] = []
    matches: list[dict[str, Any]] = []
    for typ in types:
        G = [e for e in gt_events if e["type"] == typ]
        P = [e for e in pred_events if e["type"] == typ]
        cost = np.full((len(G), len(P)), 1e6)
        for i, g in enumerate(G):
            for j, p in enumerate(P):
                dt = abs(g["t"] - p["t"])
                if dt > tol:
                    continue
                if strict and people(g, False) != people(p, True):
                    continue
                cost[i, j] = dt
        tp = 0
        errs: list[float] = []
        if G and P:
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < 1e6:
                    tp += 1
                    errs.append(cost[i, j])
                    matches.append({"type": typ, "gt_t": G[i]["t"], "pred_t": P[j]["t"], "dt": round(P[j]["t"] - G[i]["t"], 3)})
        fp, fn = len(P) - tp, len(G) - tp
        per_type[typ] = {
            "gt": len(G), "pred": len(P), "tp": tp, "fp": fp, "fn": fn,
            "mae_s": round(float(np.mean(errs)), 3) if errs else None,
        }
        tot_tp, tot_fp, tot_fn = tot_tp + tp, tot_fp + fp, tot_fn + fn
        errs_all += errs
    prec = tot_tp / (tot_tp + tot_fp) if tot_tp + tot_fp else 0.0
    rec = tot_tp / (tot_tp + tot_fn) if tot_tp + tot_fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
        "tp": tot_tp, "fp": tot_fp, "fn": tot_fn,
        "mae_s": round(float(np.mean(errs_all)), 3) if errs_all else None,
        "per_type": per_type, "matches": matches,
    }


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return the union of positive-length intervals as disjoint intervals."""
    valid = sorted((float(start), float(end)) for start, end in intervals if end > start)
    merged: list[list[float]] = []
    for start, end in valid:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _interval_length(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in _merge_intervals(intervals))


def _intersection_length(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """Measure the intersection of two interval unions without double counting."""
    left, right = _merge_intervals(a), _merge_intervals(b)
    total = 0.0
    i = j = 0
    while i < len(left) and j < len(right):
        total += max(0.0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def score_states(gt: dict[str, Any], pred_states: list[dict[str, Any]], pred2gt: dict[str, int]) -> dict[str, Any]:
    # Tag identities so an unmapped predicted ID cannot collide with a GT ID.
    PersonKey = tuple[str, int | str]

    def intervals(
        states: list[dict[str, Any]], to_gt: bool,
    ) -> dict[tuple[str, PersonKey], list[tuple[float, float]]]:
        out: dict[tuple[str, PersonKey], list[tuple[float, float]]] = {}
        for state in states:
            persons = state.get("persons", [state.get("person")])
            for person in persons:
                if person is None:
                    continue
                if to_gt:
                    mapped = pred2gt.get(str(person))
                    person_key: PersonKey = (
                        ("gt", int(mapped)) if mapped is not None else ("pred", str(person))
                    )
                else:
                    person_key = ("gt", int(person))
                out.setdefault((state["type"], person_key), []).append(
                    (state["start"], state["end"])
                )
        return out

    G, P = intervals(gt.get("states", []), False), intervals(pred_states, True)
    res: dict[str, Any] = {}
    for typ in sorted({k[0] for k in G} | {k[0] for k in P}):
        keys = {k for k in G if k[0] == typ} | {k for k in P if k[0] == typ}
        i_sum = sum(_intersection_length(G.get(k, []), P.get(k, [])) for k in keys)
        u_sum = sum(
            _interval_length(G.get(k, []))
            + _interval_length(P.get(k, []))
            - _intersection_length(G.get(k, []), P.get(k, []))
            for k in keys
        )
        tiou = i_sum / u_sum if u_sum else 0.0
        # Guard against tiny floating-point excursions outside the metric range.
        tiou = min(1.0, max(0.0, tiou))
        res[typ] = {
            "tiou": round(tiou, 3),
            "gt_people": sum(1 for k in G if k[0] == typ),
            "pred_people_mapped": sum(1 for k in P if k[0] == typ and k[1][0] == "gt"),
            "pred_people_unmapped": sum(1 for k in P if k[0] == typ and k[1][0] == "pred"),
        }
    return res


def run_benchmark(tracks: dict[str, Any], events: dict[str, Any], gt: dict[str, Any], *, tol: float = 1.0) -> dict[str, Any]:
    fps = tracks["fps"]
    gt2pred = map_gt_to_pred(gt, tracks)
    pred2gt = {v: k for k, v in gt2pred.items()}
    gt_people = gt_person_ids(gt)
    n_people = len(gt_people)

    # identity continuity: share of expected on-screen time covered by the mapped ID
    pres = gt_presence(gt)
    continuity_by_person = {person: 0.0 for person in gt_people}
    for p, gid in gt2pred.items():
        t = tracks["tracks"][gid]
        a, b = pres[p]
        covered = sum(1 for f in t["frames"] if a <= f / fps <= b) / max(1.0, (b - a) * fps)
        continuity_by_person[p] = min(1.0, covered)
    mapped_continuity = [continuity_by_person[p] for p in gt2pred]
    mapped_continuity_mean = float(np.mean(mapped_continuity)) if mapped_continuity else 0.0
    continuity_coverage = len(gt2pred) / n_people if n_people else 0.0
    long_ids = sum(1 for t in tracks["tracks"].values() if (t["frames"][-1] - t["frames"][0]) / fps >= 0.5)

    staff = {int(person) for person in gt.get("staff", [])}
    roles = {str(gid): role for gid, role in events.get("roles", {}).items()}
    mapped_role_hits = [
        (roles.get(str(gid), "") == ("staff" if p in staff else "customer"))
        for p, gid in gt2pred.items()
    ]
    role_hits = sum(mapped_role_hits)
    conditional_role_accuracy = float(np.mean(mapped_role_hits)) if mapped_role_hits else 0.0
    role_coverage = len(gt2pred) / n_people if n_people else 0.0
    staff_pred_mapped = {
        person for person, gid in gt2pred.items() if roles.get(str(gid)) == "staff"
    }
    unmapped_pred_staff = {
        gid for gid, role in roles.items() if role == "staff" and gid not in pred2gt
    }
    staff_tp = len(staff_pred_mapped & staff)

    return {
        "anchoring": {
            "mapped": len(gt2pred),
            "gt_people": n_people,
            "unmapped": sorted(gt_people - set(gt2pred)),
        },
        "identity": {
            "pred_ids": len(tracks["tracks"]), "pred_ids_ge_0.5s": long_ids, "gt_people": n_people,
            "continuity_mean": round(float(np.mean(list(continuity_by_person.values()))), 3) if gt_people else 0.0,
            "continuity_mapped": round(mapped_continuity_mean, 3),
            "continuity_mean_mapped": round(mapped_continuity_mean, 3),
            "continuity_mapped_people": len(gt2pred),
            "continuity_coverage": round(continuity_coverage, 3),
            "continuity_staff": round(float(np.mean([continuity_by_person[p] for p in staff])), 3) if staff else None,
            "continuity_staff_mapped": round(
                float(np.mean([continuity_by_person[p] for p in staff & set(gt2pred)])), 3
            ) if staff & set(gt2pred) else None,
        },
        "roles": {
            "accuracy": round(role_hits / n_people, 3) if n_people else 0.0,
            "accuracy_conditional": round(conditional_role_accuracy, 3),
            "accuracy_mapped": round(conditional_role_accuracy, 3),
            "mapped": len(gt2pred),
            "coverage": round(role_coverage, 3),
            "staff_tp": staff_tp,
            "staff_fp": len(staff_pred_mapped - staff) + len(unmapped_pred_staff),
            "staff_fn": len(staff) - staff_tp,
            "staff_unmapped_pred": len(unmapped_pred_staff),
        },
        "events_strict": score_events(gt["events"], events["events"], pred2gt, tol=tol, strict=True),
        "events_loose": score_events(gt["events"], events["events"], pred2gt, tol=tol, strict=False),
        "states": score_states(gt, events["states"], pred2gt),
        "mapping": {str(k): v for k, v in sorted(gt2pred.items())},
        "tolerance_s": tol,
    }


def print_report(r: dict[str, Any]) -> None:
    a, i, ro = r["anchoring"], r["identity"], r["roles"]
    staff_cont = "n/a" if i["continuity_staff"] is None else f"{i['continuity_staff']:.0%}"
    print(f"Anchoring  : {a['mapped']}/{a['gt_people']} GT people matched on reference frames (missing: {a['unmapped']})")
    print(f"Identity   : {i['pred_ids']} predicted IDs ({i['pred_ids_ge_0.5s']} ≥0.5s) vs {i['gt_people']} people; "
          f"continuity {i['continuity_mean']:.0%} end-to-end, {i['continuity_mapped']:.0%} mapped "
          f"(coverage {i['continuity_coverage']:.0%}, staff {staff_cont})")
    print(f"Roles      : accuracy {ro['accuracy']:.0%} end-to-end, {ro['accuracy_conditional']:.0%} mapped "
          f"(coverage {ro['coverage']:.0%}); staff TP {ro['staff_tp']} FP {ro['staff_fp']} FN {ro['staff_fn']}")
    for name in ("events_strict", "events_loose"):
        e = r[name]
        print(f"{'Events ' + name.split('_')[1]:11s}: P {e['precision']:.2f} R {e['recall']:.2f} F1 {e['f1']:.2f} "
              f"(TP {e['tp']} FP {e['fp']} FN {e['fn']}, |dt| {e['mae_s']}s, tol ±{r['tolerance_s']}s)")
    print("  per type (strict):")
    for typ, s in r["events_strict"]["per_type"].items():
        print(f"    {typ:14s} gt {s['gt']} pred {s['pred']:2d}  tp {s['tp']} fp {s['fp']:2d} fn {s['fn']}  |dt| {s['mae_s']}")
    for typ, s in r["states"].items():
        print(f"States {typ:10s}: tIoU {s['tiou']}  (gt people {s['gt_people']}, pred mapped {s['pred_people_mapped']})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tracks", type=Path, default=OUT / "tracks.json")
    p.add_argument("--events", type=Path, default=OUT / "events.json")
    p.add_argument("--gt", type=Path, default=GT_PATH)
    p.add_argument("--tol", type=float, default=1.0, help="event time tolerance, seconds")
    p.add_argument("--report", type=Path, default=OUT / "benchmark_report.json")
    args = p.parse_args()

    r = run_benchmark(
        json.loads(args.tracks.read_text()), json.loads(args.events.read_text()),
        json.loads(args.gt.read_text()), tol=args.tol,
    )
    print_report(r)
    args.report.write_text(json.dumps(r, ensure_ascii=False, indent=2))
    print(f"report → {args.report}")


if __name__ == "__main__":
    main()
