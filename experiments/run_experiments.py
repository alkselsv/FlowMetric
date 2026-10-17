#!/usr/bin/env python3
"""Detector x tracker x ReID x (local | hybrid GPT Astra) experiment grid on the benchmark clip.

Stages (each cached under data/experiments/, so re-runs only compute what is missing):
  1. detect  – person boxes for every frame per detector (conf >= 0.1) + ms/frame + anchor recall
  2. track   – BoxMOT trackers on cached detections; ultralytics BoT-SORT references for YOLO
  3. eval    – events + benchmark for local mode and for each cached Astra response
Output: benchmark/experiments.csv (every run), experiments_summary.csv (per config), experiments_tables.md
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark import run_benchmark  # noqa: E402
from events import extract_events, load_camera  # noqa: E402
from track import LocalPass, build_tracks, sync_ids  # noqa: E402

EXP = ROOT / "data" / "experiments"
GT_PATH = ROOT / "benchmark" / "ground_truth.json"
DET_CONF = 0.1
DEVICE = "mps"


def read_frames(video: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps


# ---------------------------------------------------------------------------
# Stage 1: detectors
# ---------------------------------------------------------------------------


def yolo_detector(weights: str, imgsz: int) -> Callable[[np.ndarray], np.ndarray]:
    from ultralytics import YOLO

    model = YOLO(weights)

    def detect(frame: np.ndarray) -> np.ndarray:
        r = model.predict(frame, conf=DET_CONF, iou=0.5, imgsz=imgsz, classes=[0], device=DEVICE, verbose=False)[0]
        b = r.boxes
        return np.hstack([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None]]).astype(np.float32)

    return detect


def rfdetr_detector(cls_name: str, resolution: int | None = None) -> Callable[[np.ndarray], np.ndarray]:
    import rfdetr
    from PIL import Image

    model = getattr(rfdetr, cls_name)(**({"resolution": resolution} if resolution else {}))

    def detect(frame: np.ndarray) -> np.ndarray:
        d = model.predict(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), threshold=DET_CONF)
        keep = d.class_id == 1  # person in RF-DETR's COCO-91 indexing
        return np.hstack([d.xyxy[keep], d.confidence[keep][:, None]]).astype(np.float32)

    return detect


# "+cal" entries reuse the base detector's boxes (see calibrate_to), so they have no factory.
DETECTORS: dict[str, tuple[str, Callable[[], Callable[[np.ndarray], np.ndarray]] | None]] = {
    "yolo11s@640": ("YOLO11s 640", lambda: yolo_detector("yolo11s.pt", 640)),
    "yolo11s@1280": ("YOLO11s 1280", lambda: yolo_detector("yolo11s.pt", 1280)),
    "yolo11m@1280": ("YOLO11m 1280", lambda: yolo_detector("yolo11m.pt", 1280)),
    "yolo11x@1280": ("YOLO11x 1280", lambda: yolo_detector("yolo11x.pt", 1280)),
    "yolo26s@1280": ("YOLO26s 1280", lambda: yolo_detector("yolo26s.pt", 1280)),
    "yolo26m@1280": ("YOLO26m 1280", lambda: yolo_detector("yolo26m.pt", 1280)),
    "rfdetr-s": ("RF-DETR Small", lambda: rfdetr_detector("RFDETRSmall")),
    "rfdetr-m": ("RF-DETR Medium", lambda: rfdetr_detector("RFDETRMedium")),
    "rfdetr-l": ("RF-DETR Large", lambda: rfdetr_detector("RFDETRLarge")),
    "rfdetr-m@1024": ("RF-DETR Medium 1024", lambda: rfdetr_detector("RFDETRMedium", 1024)),
    "rfdetr-l@1024": ("RF-DETR Large 1024", lambda: rfdetr_detector("RFDETRLarge", 1024)),
    "rfdetr-m+cal": ("RF-DETR Medium (calibrated)", None),
    "rfdetr-l+cal": ("RF-DETR Large (calibrated)", None),
    "rfdetr-l@1024+cal": ("RF-DETR Large 1024 (calibrated)", None),
}


def anchor_recall(dets: list[np.ndarray], gt: dict, conf: float = 0.3) -> dict[str, Any]:
    """Share of GT head anchors that fall into the upper half of some person box (one box per head)."""
    hit = total = 0
    for fi_s, heads in gt["anchors"].items():
        if not fi_s.isdigit():
            continue
        boxes = dets[int(fi_s)]
        boxes = boxes[boxes[:, 4] >= conf]
        used: set[int] = set()
        for _, (hx, hy) in heads.items():
            total += 1
            best, best_d = None, 1e9
            for j, (x1, y1, x2, y2, _) in enumerate(boxes):
                if j in used or not (x1 <= hx <= x2 and y1 <= hy <= y1 + 0.5 * (y2 - y1)):
                    continue
                d = abs(hx - (x1 + x2) / 2) + abs(hy - y1)
                if d < best_d:
                    best, best_d = j, d
            if best is not None:
                used.add(best)
                hit += 1
    return {"anchor_hits": hit, "anchor_total": total}


CAL_REF = "yolo11s@1280"


def calibrate_to(dets: list[np.ndarray], ref: list[np.ndarray]) -> list[np.ndarray]:
    """Rank-based (monotone) remap of confidences onto the reference detector's score distribution.

    Box order is preserved; only the scale changes, so shared tracker thresholds mean the same for both.
    """
    src = np.concatenate([d[:, 4] for d in dets])
    ref_sorted = np.sort(np.concatenate([d[:, 4] for d in ref]))[::-1]
    ranks = np.empty(len(src), int)
    ranks[np.argsort(-src, kind="stable")] = np.arange(len(src))
    mapped = np.where(ranks < len(ref_sorted), ref_sorted[np.minimum(ranks, len(ref_sorted) - 1)], DET_CONF * 0.99)
    out, i = [], 0
    for d in dets:
        c = d.copy()
        c[:, 4] = mapped[i:i + len(d)]
        out.append(c)
        i += len(d)
    return out


def stage_detect(frames: list[np.ndarray], names: list[str], gt: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name in names:
        path = EXP / "dets" / f"{name}.pkl"
        if path.exists():
            out[name] = pickle.loads(path.read_bytes())
            continue
        if name.endswith("+cal"):
            base, ref = out.get(name[:-4]), out.get(CAL_REF)
            if base is None or ref is None:
                print(f"[detect] {name}: needs {name[:-4]} and {CAL_REF} first", flush=True)
                continue
            dets = calibrate_to(base["dets"], ref["dets"])
            out[name] = {"dets": dets, "ms": base["ms"], **anchor_recall(dets, gt)}
            path.write_bytes(pickle.dumps(out[name]))
            print(f"[detect] {name}: anchors {out[name]['anchor_hits']}/{out[name]['anchor_total']}", flush=True)
            continue
        print(f"[detect] {name}", flush=True)
        try:
            detect = DETECTORS[name][1]()
            for f in frames[:3]:  # warm-up
                detect(f)
            t0 = time.perf_counter()
            dets = [detect(f) for f in frames]
            ms = (time.perf_counter() - t0) / len(frames) * 1000
        except Exception as e:
            print(f"[detect] {name} FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        rec = {"dets": dets, "ms": ms, **anchor_recall(dets, gt)}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(rec))
        print(f"[detect] {name}: {ms:.0f} ms/frame, anchors {rec['anchor_hits']}/{rec['anchor_total']}", flush=True)
        out[name] = rec
    return out


# ---------------------------------------------------------------------------
# Stage 2: trackers
# ---------------------------------------------------------------------------


class _NoCMC:
    """Static camera: identity warp (also keeps tracker cost comparable)."""

    def apply(self, img, dets=None):
        return np.eye(2, 3)


def _patch_boxmot_numpy2() -> None:
    """BoxMOT 12 OC-SORT/Deep OC-SORT re-update calls float() on (1,)-arrays, which numpy >= 2.5 rejects."""
    from collections import deque

    from boxmot.motion.kalman_filters.aabb.xysr_kf import KalmanFilterXYSR as KF

    if getattr(KF, "_np2_patched", False):
        return
    orig = KF.unfreeze

    def unfreeze(self):
        if self.attr_saved is not None:
            self.history_obs = deque(
                [None if b is None else np.asarray(b, float).reshape(-1) for b in self.history_obs],
                maxlen=self.history_obs.maxlen,
            )
        return orig(self)

    KF.unfreeze = unfreeze
    KF._np2_patched = True


def boxmot_tracker(name: str, fps: float):
    import boxmot

    _patch_boxmot_numpy2()

    w = {"osnet025": "osnet_x0_25_msmt17.pt", "osnet10": "osnet_x1_0_msmt17.pt", "clip": "clip_market1501.pt"}
    common = {"device": DEVICE, "half": False}
    fr = int(round(fps))
    if name == "bytetrack":
        return boxmot.ByteTrack(min_conf=DET_CONF, track_thresh=0.3, match_thresh=0.8, track_buffer=30, frame_rate=fr)
    if name == "ocsort":
        return boxmot.OcSort(min_conf=DET_CONF, det_thresh=0.3, max_age=30, min_hits=3)
    if name.startswith("botsort"):
        reid = name.split("+")[1] if "+" in name else None
        t = boxmot.BotSort(
            reid_weights=Path(w[reid or "osnet025"]), **common, track_high_thresh=0.3, track_low_thresh=DET_CONF,
            new_track_thresh=0.4, track_buffer=30, frame_rate=fr, cmc_method="sof", with_reid=reid is not None,
        )
        t.cmc = _NoCMC()
        return t
    if name.startswith("deepocsort"):
        return boxmot.DeepOcSort(reid_weights=Path(w[name.split("+")[1]]), **common, det_thresh=0.3,
                                 max_age=30, min_hits=3, cmc_off=True)
    if name.startswith("strongsort"):
        t = boxmot.StrongSort(reid_weights=Path(w[name.split("+")[1]]), **common, min_conf=0.3, max_age=30, n_init=3)
        t.cmc = _NoCMC()
        return t
    if name.startswith("boosttrack"):
        return boxmot.BoostTrack(reid_weights=Path(w[name.split("+")[1]]), **common, det_thresh=0.3,
                                 max_age=30, min_hits=3, use_ecc=False, with_reid=True)
    raise ValueError(name)


BOXMOT_TRACKERS = {
    "bytetrack": ("ByteTrack", "—"),
    "ocsort": ("OC-SORT", "—"),
    "botsort": ("BoT-SORT", "—"),
    "botsort+osnet025": ("BoT-SORT", "OSNet x0.25 MSMT17"),
    "botsort+osnet10": ("BoT-SORT", "OSNet x1.0 MSMT17"),
    "botsort+clip": ("BoT-SORT", "CLIP-ReID Market1501"),
    "deepocsort+osnet025": ("Deep OC-SORT", "OSNet x0.25 MSMT17"),
    "strongsort+osnet025": ("StrongSORT", "OSNet x0.25 MSMT17"),
    "boosttrack+osnet025": ("BoostTrack++", "OSNet x0.25 MSMT17"),
}
UL_TRACKERS = {
    "ul-botsort": ("BoT-SORT (ultralytics, prod)", "—", "botsort.yaml"),
    "ul-botsort+auto": ("BoT-SORT (ultralytics)", "YOLO features (model: auto)", str(ROOT / "config" / "botsort_reid.yaml")),
}


def _lp(frames: list[np.ndarray], fps: float) -> LocalPass:
    h, w = frames[0].shape[:2]
    return LocalPass(fps=fps, width=w, height=h, total=len(frames))


WARMUP_FRAMES = 3  # >= n_init/min_hits, so confirmation-delayed trackers report people on frame 0 (GT anchor)


def run_boxmot(frames: list[np.ndarray], fps: float, dets: list[np.ndarray], trk: str) -> tuple[LocalPass, float]:
    tracker = boxmot_tracker(trk, fps)
    lp = _lp(frames, fps)

    def d6(d: np.ndarray) -> np.ndarray:
        return np.hstack([d, np.zeros((len(d), 1), np.float32)]) if len(d) else np.empty((0, 6), np.float32)

    for _ in range(WARMUP_FRAMES):  # static camera: as if the stream had been running before frame 0
        tracker.update(d6(dets[0]), frames[0])
    t0 = time.perf_counter()
    for fi, (frame, d) in enumerate(zip(frames, dets)):
        out = tracker.update(d6(d), frame)
        lp.dets[fi] = [(int(r[4]), r[:4].astype(float), float(r[5])) for r in np.asarray(out).reshape(-1, 8)]
    return lp, (time.perf_counter() - t0) / len(frames) * 1000


def run_ultralytics(frames: list[np.ndarray], fps: float, weights: str, imgsz: int, cfg: str) -> tuple[LocalPass, float]:
    from ultralytics import YOLO

    model = YOLO(weights)
    lp = _lp(frames, fps)
    t0 = time.perf_counter()
    for fi, frame in enumerate(frames):
        r = model.track(frame, persist=True, conf=0.2, iou=0.5, imgsz=imgsz, classes=[0], tracker=cfg,
                        device=DEVICE, verbose=False)[0]
        b = r.boxes
        lp.dets[fi] = [] if b is None or b.id is None else [
            (int(t), x.astype(float), float(c))
            for x, t, c in zip(b.xyxy.cpu().numpy(), b.id.cpu().numpy().astype(int), b.conf.cpu().numpy())
        ]
    return lp, (time.perf_counter() - t0) / len(frames) * 1000


def stage_track(frames, fps, det_cache: dict[str, dict], trackers: list[str], ul: bool, compute: bool) -> list[dict]:
    runs = []
    for det_name, det in det_cache.items():
        jobs = list(trackers)
        if ul and det_name.startswith("yolo"):
            jobs += list(UL_TRACKERS)
        for trk in jobs:
            tag = "" if trk in UL_TRACKERS else f".w{WARMUP_FRAMES}"
            path = EXP / "tracks" / f"{det_name}__{trk}{tag}.pkl"
            if not path.exists():
                if not compute:
                    continue
                print(f"[track] {det_name} + {trk}", flush=True)
                try:
                    if trk in UL_TRACKERS:
                        weights, imgsz = det_name.split("@")
                        lp, ms = run_ultralytics(frames, fps, f"{weights}.pt", int(imgsz), UL_TRACKERS[trk][2])
                        ms_trk = max(ms - det["ms"], 0.0)  # model.track includes detection
                    else:
                        lp, ms_trk = run_boxmot(frames, fps, det["dets"], trk)
                except Exception as e:
                    print(f"[track] {det_name} + {trk} FAILED: {type(e).__name__}: {e}", flush=True)
                    traceback.print_exc()
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(pickle.dumps({"lp_dets": lp.dets, "fps": lp.fps, "w": lp.width, "h": lp.height,
                                               "n": lp.total, "ms_trk": ms_trk}))
                print(f"[track] {det_name} + {trk}: {ms_trk:.0f} ms/frame", flush=True)
            runs.append({"det": det_name, "trk": trk, "path": path})
    return runs


# ---------------------------------------------------------------------------
# Stage 3: evaluation
# ---------------------------------------------------------------------------


def load_astra() -> dict[str, dict]:
    out = {}
    for p in sorted((EXP / "astra").glob("*.json")):
        out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    return out


def evaluate(run: dict, det_cache: dict, astra: dict, gt: dict, cam: dict) -> list[dict]:
    rec = pickle.loads(run["path"].read_bytes())
    lp = LocalPass(fps=rec["fps"], width=rec["w"], height=rec["h"], total=rec["n"], dets=rec["lp_dets"])
    det = det_cache[run["det"]]
    if run["trk"] in UL_TRACKERS:
        trk_label, reid_label, _ = UL_TRACKERS[run["trk"]]
    else:
        trk_label, reid_label = BOXMOT_TRACKERS[run["trk"]]
    base = {
        "detector": DETECTORS[run["det"]][0], "det_key": run["det"], "tracker": trk_label, "trk_key": run["trk"],
        "reid": reid_label, "det_ms": round(det["ms"], 1), "trk_ms": round(rec["ms_trk"], 1),
        "anchor_recall": round(det["anchor_hits"] / det["anchor_total"], 3),
        "local_tracks": len({t for d in lp.dets.values() for t, _, _ in d}),
    }
    modes: list[tuple[str, dict | None]] = [("local", None)] + [(k, v) for k, v in astra.items()]
    rows = []
    for mode, a in modes:
        if a is None:
            tid_to_gid, gid_role, stats = {}, {}, {}
        else:
            tid_to_gid, gid_role, stats = sync_ids(lp, a["reids"], iou_thr=0.3)
        tracks = build_tracks(lp, tid_to_gid, gid_role, mode="local" if a is None else "hybrid")
        ev = extract_events(tracks, cam)
        b = run_benchmark(tracks, ev, gt)
        es, el, st = b["events_strict"], b["events_loose"], b["states"]
        rows.append({
            **base,
            "mode": mode,
            "astra_calls": a["meta"]["calls"] if a else 0,
            "astra_frames": a["meta"]["frames"] if a else 0,
            "pred_ids": b["identity"]["pred_ids"],
            "anchored": b["anchoring"]["mapped"],
            "continuity": b["identity"]["continuity_mean"],
            "continuity_staff": b["identity"]["continuity_staff"],
            "role_acc": b["roles"]["accuracy"],
            "strict_p": es["precision"], "strict_r": es["recall"], "strict_f1": es["f1"],
            "strict_tp": es["tp"], "strict_fp": es["fp"], "strict_fn": es["fn"], "mae_s": es["mae_s"],
            "loose_f1": el["f1"],
            "waiting_tiou": (st.get("waiting") or {}).get("tiou"),
            "consuming_tiou": (st.get("consuming") or {}).get("tiou"),
            "astra_verified": stats.get("astra_verified"),
        })
    return rows


def fmt(v: Any, pct: bool = False) -> str:
    if v is None:
        return "—"
    if pct:
        return f"{v * 100:.0f}%"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def write_tables(rows: list[dict], det_cache: dict, path: Path) -> None:
    lines = ["# Experiment tables (auto-generated by experiments/run_experiments.py)", ""]
    lines += ["## Detectors", "", "| Detector | ms/frame (MPS) | GT heads found (conf ≥ 0.3) |", "|---|---:|---:|"]
    for k, d in det_cache.items():
        lines.append(f"| {DETECTORS[k][0]} | {d['ms']:.0f} | {d['anchor_hits']}/{d['anchor_total']} |")
    hdr = ("| Detector | Tracker | ReID | Mode | IDs | Continuity | Staff cont. | Roles | Strict P | Strict R | "
           "Strict F1 | Loose F1 | Waiting tIoU | Consuming tIoU | Track ms |")
    sep = "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    for title, flt in [("Local only", lambda r: r["mode"] == "local"),
                       ("Hybrid (local + GPT-6 Astra 1 fps)", lambda r: r["mode"] == "astra_1fps"),
                       ("All hybrid variants", lambda r: r["mode"] not in ("local", "astra_1fps"))]:
        sel = sorted((r for r in rows if flt(r)), key=lambda r: (-r["strict_f1"], -r["continuity"]))
        if not sel:
            continue
        lines += ["", f"## {title}", "", hdr, sep]
        for r in sel:
            lines.append("| " + " | ".join([
                r["detector"], r["tracker"], r["reid"], r["mode"], str(r["pred_ids"]), fmt(r["continuity"], True),
                fmt(r["continuity_staff"], True), fmt(r["role_acc"], True), fmt(r["strict_p"]), fmt(r["strict_r"]),
                fmt(r["strict_f1"]), fmt(r["loose_f1"]), fmt(r["waiting_tiou"]), fmt(r["consuming_tiou"]),
                f"{r['trk_ms']:.0f}",
            ]) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


HYBRID_REPEATS = ("astra_1fps", "astra_1fps_r2", "astra_1fps_r3", "astra_1fps_medium")


def write_summary(rows: list[dict], path: Path) -> None:
    """One row per detector+tracker: local metrics and hybrid metrics averaged over repeated Astra 1 fps answers."""
    by_cfg: dict[tuple[str, str], dict[str, list[dict]]] = {}
    for r in rows:
        by_cfg.setdefault((r["det_key"], r["trk_key"]), {}).setdefault(r["mode"], []).append(r)
    out = []
    for (det, trk), modes in by_cfg.items():
        loc = modes.get("local", [None])[0]
        hyb = [r for m in HYBRID_REPEATS for r in modes.get(m, [])]
        if loc is None:
            continue
        rec = {k: loc[k] for k in ("detector", "tracker", "reid", "det_ms", "trk_ms", "anchor_recall")}
        rec |= {f"local_{k}": loc[k] for k in ("pred_ids", "continuity", "continuity_staff", "role_acc", "strict_p",
                                              "strict_r", "strict_f1", "waiting_tiou", "consuming_tiou")}
        for k in ("pred_ids", "continuity", "continuity_staff", "role_acc", "strict_p", "strict_r", "strict_f1",
                  "waiting_tiou", "consuming_tiou"):
            vals = [r[k] for r in hyb if r[k] is not None]
            rec[f"hybrid_{k}"] = round(float(np.mean(vals)), 3) if vals else None
        f1 = [r["strict_f1"] for r in hyb]
        rec["hybrid_strict_f1_min"], rec["hybrid_strict_f1_max"] = (min(f1), max(f1)) if f1 else (None, None)
        rec["hybrid_runs"] = len(hyb)
        out.append(rec)
    out.sort(key=lambda r: (-(r["hybrid_strict_f1"] or 0), -(r["hybrid_continuity"] or 0)))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, default=next((ROOT / "data").glob("*.mp4")))
    p.add_argument("--detectors", nargs="*", default=list(DETECTORS))
    p.add_argument("--trackers", nargs="*", default=list(BOXMOT_TRACKERS))
    p.add_argument("--no-ul", action="store_true", help="skip ultralytics BoT-SORT reference runs")
    p.add_argument("--stages", default="detect,track,eval")
    args = p.parse_args()

    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    cam = load_camera(None)
    frames, fps = read_frames(args.video)
    print(f"video: {len(frames)} frames @ {fps:.1f} fps", flush=True)

    det_cache = stage_detect(frames, args.detectors, gt)
    runs = stage_track(frames, fps, det_cache, args.trackers, ul=not args.no_ul, compute="track" in args.stages)
    if "eval" not in args.stages:
        return

    astra = load_astra()
    print(f"[eval] {len(runs)} tracker runs x modes local + {list(astra)}", flush=True)
    rows = []
    for run in runs:
        try:
            rows += evaluate(run, det_cache, astra, gt, cam)
        except Exception as e:
            print(f"[eval] {run['det']} + {run['trk']} FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

    out_csv = ROOT / "benchmark" / "experiments.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_tables(rows, det_cache, ROOT / "benchmark" / "experiments_tables.md")
    write_summary(rows, ROOT / "benchmark" / "experiments_summary.csv")
    print(f"[eval] {len(rows)} rows → {out_csv}", flush=True)


if __name__ == "__main__":
    main()
