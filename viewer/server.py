"""Small dependency-free HTTP server for the FlowMetric viewer."""

from __future__ import annotations

import json
import errno
import mimetypes
import re
import shutil
import subprocess
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STATIC_ROOT = Path(__file__).resolve().parent / "static"
BROWSER_VIDEO_CODECS = {"h264", "vp8", "vp9", "av1"}
PORT_FALLBACK_ATTEMPTS = 20


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return {}
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _run_score(report: dict[str, Any]) -> float:
    """Rank completed runs by identity continuity and strict event quality."""
    identity = report.get("identity", {})
    continuity = float(
        identity.get("continuity_mapped", identity.get("continuity_mean_mapped", identity.get("continuity_mean", 0)))
    )
    staff = float(identity.get("continuity_staff_mapped", identity.get("continuity_staff", 0)))
    coverage = float(identity.get("continuity_coverage", 1.0))
    event_f1 = float(report.get("events_strict", {}).get("f1", 0))
    return 0.45 * continuity + 0.25 * staff + 0.25 * event_f1 + 0.05 * coverage


def select_best_run(output_root: Path) -> Path:
    """Select the highest-quality complete viewer run under data/output."""
    candidates: list[tuple[float, float, Path]] = []
    for report_path in output_root.rglob("benchmark_report.json"):
        run = report_path.parent
        if not all((run / name).is_file() for name in ("tracks.json", "events.json", "tracked.mp4")):
            continue
        try:
            score = _run_score(_read_json(report_path))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        candidates.append((score, (run / "tracks.json").stat().st_mtime, run))
    if candidates:
        return max(candidates, key=lambda item: (item[0], item[1]))[2]
    if all((output_root / name).is_file() for name in ("tracks.json", "events.json", "tracked.mp4")):
        return output_root
    raise FileNotFoundError(f"no complete FlowMetric runs found under {output_root}")


def _browser_video(path: Path) -> Path:
    """Return a browser-compatible video, transcoding legacy mp4v output once."""
    if not path.is_file():
        return path
    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    if not ffprobe or not ffmpeg:
        return path
    probe = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode == 0 and probe.stdout.strip() in BROWSER_VIDEO_CODECS:
        return path
    converted = path.with_name(f"{path.stem}.viewer.mp4")
    if converted.is_file() and converted.stat().st_mtime >= path.stat().st_mtime:
        return converted
    print(f"Converting {path.name} to browser-compatible H.264…")
    result = subprocess.run(
        [
            ffmpeg, "-y", "-loglevel", "error", "-i", str(path), "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(converted),
        ],
        check=False,
    )
    if result.returncode != 0 or not converted.is_file():
        print("Warning: video transcoding failed; serving the original codec.")
        return path
    return converted


def _homography(source: list[list[float]], target: list[list[float]]) -> np.ndarray:
    """Return a projective transform for four or more corresponding points."""
    if len(source) != len(target) or len(source) < 4:
        raise ValueError("floorplan source_points and target_points need 4+ pairs")
    rows: list[list[float]] = []
    values: list[float] = []
    for (x, y), (u, v) in zip(source, target):
        rows.extend(
            ([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y],
             [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        )
        values.extend((u, v))
    solution, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(values), rcond=None)
    return np.append(solution, 1.0).reshape(3, 3)


def _projector(
    width: float, height: float, floorplan: dict[str, Any]
) -> tuple[Callable[[float, float], list[float]], dict[str, Any]]:
    if not floorplan:
        def normalized(x: float, y: float) -> list[float]:
            return [round(x / width, 6), round(y / height, 6)]

        return normalized, {"mode": "camera_plane", "label": "Плоскость камеры"}

    source = floorplan.get("source_points")
    target = floorplan.get("target_points")
    size = floorplan.get("size")
    if not (
        isinstance(source, list)
        and isinstance(target, list)
        and isinstance(size, list)
        and len(size) == 2
        and float(size[0]) > 0
        and float(size[1]) > 0
    ):
        raise ValueError(
            "floorplan needs source_points, target_points, and positive size [width, height]"
        )
    matrix = _homography(source, target)
    floor_width, floor_height = map(float, size)

    def calibrated(x: float, y: float) -> list[float]:
        raw = matrix @ np.asarray([x, y, 1.0])
        if abs(raw[2]) < 1e-9:
            return [0.0, 0.0]
        return [round(float(raw[0] / raw[2] / floor_width), 6),
                round(float(raw[1] / raw[2] / floor_height), 6)]

    return calibrated, {
        "mode": "floorplan",
        "label": "План зала",
        "size": [floor_width, floor_height],
        "units": floorplan.get("units", "m"),
    }


def _polygon(points: Any, project: Callable[[float, float], list[float]]) -> list[list[float]]:
    if not isinstance(points, list):
        return []
    return [project(float(point[0]), float(point[1])) for point in points]


def _service_intervals(events_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Use audited intervals when present, otherwise pair legacy point events."""
    explicit = events_data.get("service_intervals")
    if isinstance(explicit, list) and explicit:
        return explicit
    starts: dict[tuple[str, str], list[float]] = {}
    inferred: list[dict[str, Any]] = []
    for event in sorted(events_data.get("events", []), key=lambda item: item.get("t", 0)):
        if event.get("type") not in {"service_start", "service_end"}:
            continue
        key = (str(event.get("staff")), str(event.get("customer")))
        if event["type"] == "service_start":
            starts.setdefault(key, []).append(float(event["t"]))
        elif starts.get(key):
            inferred.append(
                {
                    "staff": key[0],
                    "customer": key[1],
                    "start": starts[key].pop(0),
                    "end": float(event["t"]),
                    "source": "point_events",
                }
            )
    return inferred


def _quality_metrics(report: dict[str, Any]) -> dict[str, float]:
    """Keep only viewer-facing benchmark metrics from a potentially large report."""
    if not report:
        return {}
    identity = report.get("identity", {})
    roles = report.get("roles", {})
    events = report.get("events_strict", {})
    states = report.get("states", {})
    waiting = states.get("waiting", {}) if isinstance(states, dict) else {}
    metrics = {
        "identity_continuity": identity.get(
            "continuity_mean",
            identity.get("continuity_mapped", identity.get("continuity_mean_mapped")),
        ),
        "role_accuracy": roles.get("accuracy", roles.get("accuracy_mapped")),
        "event_f1": events.get("f1"),
        "waiting_tiou": waiting.get("tiou"),
    }
    return {key: float(value) for key, value in metrics.items() if value is not None}


def build_session(
    tracks_path: Path,
    events_path: Path,
    tables_path: Path | None = None,
    camera_path: Path | None = None,
    floorplan_path: Path | None = None,
    table_vacancy_grace_seconds: float = 2.0,
) -> dict[str, Any]:
    """Load pipeline artifacts and convert image positions into viewer coordinates."""
    tracks_data = _read_json(tracks_path)
    events_data = _read_json(events_path)
    tables_data = _read_json(tables_path, required=False) if tables_path else {}
    camera_data = _read_json(camera_path, required=False) if camera_path else {}
    floorplan_data = _read_json(floorplan_path) if floorplan_path else {}
    quality = _quality_metrics(
        _read_json(tracks_path.parent / "benchmark_report.json", required=False)
    )

    fps = float(tracks_data.get("fps", 0))
    width = float(tracks_data.get("width", 0))
    height = float(tracks_data.get("height", 0))
    n_frames = int(tracks_data.get("n_frames", 0))
    if fps <= 0 or width <= 0 or height <= 0 or n_frames <= 0:
        raise ValueError("tracks.json needs positive fps, width, height, and n_frames")
    table_resolution = tables_data.get("resolution")
    if (
        isinstance(table_resolution, list)
        and len(table_resolution) == 2
        and (float(table_resolution[0]) != width or float(table_resolution[1]) != height)
    ):
        tables_data = {}
    raw_tracks = tracks_data.get("tracks")
    if not isinstance(raw_tracks, dict):
        raise ValueError("tracks.json field 'tracks' must be an object")

    project, projection = _projector(width, height, floorplan_data)
    roles = events_data.get("roles", {})
    output_tracks: list[dict[str, Any]] = []
    for track_id, track in raw_tracks.items():
        frames = track.get("frames", [])
        boxes = track.get("boxes", [])
        confidence = track.get("conf", [])
        if len(frames) != len(boxes):
            raise ValueError(f"track {track_id} has different frame and box counts")
        points = []
        for index, (frame, box) in enumerate(zip(frames, boxes)):
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError(f"track {track_id} contains an invalid box")
            x = (float(box[0]) + float(box[2])) / 2
            y = float(box[3])
            point = project(x, y)
            sample = [round(float(frame) / fps, 3), *point]
            if index < len(confidence):
                sample.append(round(float(confidence[index]), 3))
            points.append(sample)
        output_tracks.append(
            {
                "id": str(track_id),
                "role": roles.get(str(track_id), track.get("role", "")),
                "points": points,
            }
        )

    tables: list[dict[str, Any]] = []
    floor_tables = floorplan_data.get("tables") if floorplan_data else None
    if isinstance(floor_tables, list):
        floor_width, floor_height = projection["size"]
        for index, table in enumerate(floor_tables):
            raw = table.get("polygon", [])
            polygon = [[round(float(x) / floor_width, 6), round(float(y) / floor_height, 6)] for x, y in raw]
            tables.append({"id": str(table.get("id", index + 1)), "polygon": polygon})
    else:
        for index, table in enumerate(tables_data.get("tables", [])):
            box = table.get("box", [])
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = map(float, box)
            tables.append(
                {
                    "id": str(table.get("id", index + 1)),
                    "polygon": [project(x1, y1), project(x2, y1), project(x2, y2), project(x1, y2)],
                    "support": table.get("support"),
                }
            )

    zones = []
    for name in ("door", "kitchen"):
        points = _polygon(camera_data.get(name), project)
        if points:
            zones.append({"id": name, "polygon": points})

    events = events_data.get("events", [])
    states = events_data.get("states", [])
    if not isinstance(events, list) or not isinstance(states, list):
        raise ValueError("events.json fields 'events' and 'states' must be arrays")
    events = [dict(event) for event in events]
    raw_table_items = tables_data.get("tables", [])
    raw_table_boxes = [item.get("box", []) if isinstance(item, dict) else item for item in raw_table_items]

    def table_index_at_box(box: Any) -> int | None:
        if not isinstance(box, list) or len(box) != 4:
            return None
        height = float(box[3]) - float(box[1])
        lap_x = (float(box[0]) + float(box[2])) / 2
        lap_y = float(box[1]) + 0.6 * height
        margin = 0.3 * height
        best: tuple[float, int] | None = None
        for table_index, table_box in enumerate(raw_table_boxes):
            if not isinstance(table_box, list) or len(table_box) != 4:
                continue
            x1, y1, x2, y2 = map(float, table_box)
            if not (x1 - margin <= lap_x <= x2 + margin and y1 - margin <= lap_y <= y2 + margin):
                continue
            diagonal = max(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5, 1.0)
            distance = (((lap_x - (x1 + x2) / 2) ** 2 + (lap_y - (y1 + y2) / 2) ** 2) ** 0.5) / diagonal
            if best is None or distance < best[0]:
                best = (distance, table_index)
        return None if best is None else best[1]

    seated_at: dict[str, float] = {}
    for event in events:
        if event.get("type") == "seated" and event.get("person") is not None:
            person = str(event["person"])
            seated_at[person] = min(seated_at.get(person, float("inf")), float(event.get("t", 0)))
    customer_tables: dict[str, str] = {}
    for person, track in raw_tracks.items():
        person = str(person)
        if roles.get(person, track.get("role", "")) != "customer":
            continue
        frames = track.get("frames", [])
        boxes = track.get("boxes", [])
        if not frames or len(frames) != len(boxes) or not raw_table_boxes:
            continue
        target_frame = seated_at.get(person, float(frames[0]) / fps) * fps
        start_index = min(range(len(frames)), key=lambda item: abs(float(frames[item]) - target_frame))
        votes: dict[int, int] = {}
        for box in boxes[start_index:]:
            table_index = table_index_at_box(box)
            if table_index is not None:
                votes[table_index] = votes.get(table_index, 0) + 1
        if not votes:
            continue
        best_index = max(votes, key=votes.get)
        item = raw_table_items[best_index]
        customer_tables[person] = str(
            item.get("id", best_index + 1) if isinstance(item, dict) else best_index + 1
        )

    for event in events:
        if event.get("type") not in {"seated", "service_start", "service_end"}:
            continue
        person = str(event.get("customer", event.get("person", "")))
        table_id = customer_tables.get(person)
        if table_id is not None:
            event["table"] = table_id

    analysis_end = round((n_frames - 1) / fps, 3)
    raw_occupancy: dict[str, list[tuple[float, float, str]]] = {}
    customers_with_state: set[str] = set()
    for item in states:
        if item.get("type") not in {"waiting", "consuming"}:
            continue
        person = str(item.get("person", ""))
        table_id = customer_tables.get(person)
        if table_id is None:
            continue
        start = max(0.0, float(item.get("start", 0)))
        end = min(analysis_end, float(item.get("end", analysis_end)))
        if end < start:
            continue
        raw_occupancy.setdefault(table_id, []).append((start, end, person))
        customers_with_state.add(person)
    for person, start in seated_at.items():
        if person in customers_with_state or person not in customer_tables:
            continue
        track = raw_tracks.get(person, {})
        frames = track.get("frames", [])
        if frames:
            end = min(analysis_end, float(frames[-1]) / fps)
            if end >= start:
                raw_occupancy.setdefault(customer_tables[person], []).append((start, end, person))

    grace = max(0.0, float(table_vacancy_grace_seconds))
    table_occupancy: list[dict[str, Any]] = []
    for table_id, intervals in raw_occupancy.items():
        intervals.sort(key=lambda item: (item[0], item[1]))
        current_start, current_end, first_person = intervals[0]
        customers = {first_person}
        for start, end, person in intervals[1:]:
            if start <= current_end + grace:
                current_end = max(current_end, end)
                customers.add(person)
                continue
            release_end = current_end + grace
            right_censored = release_end >= analysis_end
            table_occupancy.append(
                {
                    "table": table_id,
                    "start": round(current_start, 3),
                    "end": round(analysis_end if right_censored else release_end, 3),
                    "customers": sorted(customers),
                    "left_censored": current_start <= 0,
                    "right_censored": right_censored,
                }
            )
            current_start, current_end, customers = start, end, {person}
        release_end = current_end + grace
        right_censored = release_end >= analysis_end
        table_occupancy.append(
            {
                "table": table_id,
                "start": round(current_start, 3),
                "end": round(analysis_end if right_censored else release_end, 3),
                "customers": sorted(customers),
                "left_censored": current_start <= 0,
                "right_censored": right_censored,
            }
        )
    table_occupancy.sort(key=lambda item: (item["start"], item["table"]))
    return {
        "meta": {
            "run": tracks_path.parent.name,
            "mode": tracks_data.get("mode", ""),
            "fps": fps,
            "width": width,
            "height": height,
            "n_frames": n_frames,
            "duration": analysis_end,
            "projection": projection,
            "table_vacancy_grace_seconds": grace,
        },
        "tracks": output_tracks,
        "tables": tables,
        "customer_tables": customer_tables,
        "table_occupancy": table_occupancy,
        "zones": zones,
        "events": events,
        "states": states,
        "service_intervals": _service_intervals(events_data),
        "service_candidates": events_data.get("service_candidates", []),
        "quality": quality,
    }


class ViewerHandler(SimpleHTTPRequestHandler):
    session: dict[str, Any] = {}
    video_path: Path | None = None

    def _video_range(self, *, send_body: bool) -> None:
        path = self.video_path
        if path is None or not path.is_file():
            self.send_error(404, "Video not found")
            return
        size = path.stat().st_size
        range_header = self.headers.get("Range")
        start, end = 0, size - 1
        status = 200
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match:
                self.send_error(416, "Invalid byte range")
                return
            raw_start, raw_end = match.groups()
            if not raw_start and not raw_end:
                self.send_error(416, "Invalid byte range")
                return
            if raw_start:
                start = int(raw_start)
                end = int(raw_end) if raw_end else size - 1
            else:
                suffix = int(raw_end)
                start = max(0, size - suffix)
            if start >= size or end < start:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            end = min(end, size - 1)
            status = 206
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if not send_body:
            return
        with path.open("rb") as source:
            source.seek(start)
            remaining = length
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_GET(self) -> None:  # noqa: N802 - stdlib hook
        if self.path.rstrip("/") == "/api/video":
            self._video_range(send_body=True)
            return
        if self.path.rstrip("/") == "/api/session":
            body = json.dumps(self.session, ensure_ascii=False, separators=(",", ":")).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib hook
        if self.path.rstrip("/") == "/api/video":
            self._video_range(send_body=False)
            return
        super().do_HEAD()

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        if self.path == "/" or self.path.startswith(("/index.html", "/app.js", "/styles.css")):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:
        print(f"viewer: {format % args}")


def _bind_server(
    host: str,
    port: int,
    handler: Callable[..., SimpleHTTPRequestHandler],
) -> ThreadingHTTPServer:
    """Bind the requested port, falling forward when another viewer uses it."""
    if port == 0:
        return ThreadingHTTPServer((host, 0), handler)
    for candidate in range(port, port + PORT_FALLBACK_ATTEMPTS + 1):
        try:
            server = ThreadingHTTPServer((host, candidate), handler)
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                continue
            raise
        if candidate != port:
            print(f"Port {port} is busy; using {candidate} instead.")
        return server
    raise OSError(
        errno.EADDRINUSE,
        f"ports {port}-{port + PORT_FALLBACK_ATTEMPTS} are already in use",
    )


def run_server(
    *,
    tracks_path: Path,
    events_path: Path,
    video_path: Path,
    host: str,
    port: int,
    open_browser: bool,
    waiting_sla_seconds: float = 180.0,
    table_vacancy_grace_seconds: float = 2.0,
) -> None:
    tables_path = ROOT / "config" / "tables.json"
    camera_path = ROOT / "config" / "camera.json"
    session = build_session(
        tracks_path,
        events_path,
        tables_path=tables_path if tables_path.is_file() else None,
        camera_path=camera_path if camera_path.is_file() else None,
        table_vacancy_grace_seconds=table_vacancy_grace_seconds,
    )
    served_video = _browser_video(video_path)
    session["meta"]["duration"] = round(
        float(session["meta"]["n_frames"]) / float(session["meta"]["fps"]), 3
    )
    session["meta"]["waiting_sla_seconds"] = float(waiting_sla_seconds)
    session["meta"]["video"] = {
        "available": served_video.is_file(),
        "url": "/api/video" if served_video.is_file() else None,
        "name": video_path.name,
    }
    mimetypes.add_type("text/javascript", ".js")
    handler = partial(ViewerHandler, directory=str(STATIC_ROOT))
    ViewerHandler.session = session
    ViewerHandler.video_path = served_video
    server = _bind_server(host, port, handler)
    url = f"http://{host}:{server.server_port}"
    print(f"FlowMetric viewer: {url}")
    print("Press Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.35, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
