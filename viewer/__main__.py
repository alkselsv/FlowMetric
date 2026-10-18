from __future__ import annotations

import argparse
from pathlib import Path

from .server import run_server, select_best_run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open an interactive FlowMetric trajectory and event viewer."
    )
    parser.add_argument("--run", type=Path, help="Run directory. By default the best benchmarked run is selected.")
    parser.add_argument("--tracks", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Preferred port (default: 8765); the next free port is used if busy.",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not open the viewer in a browser."
    )
    parser.add_argument(
        "--waiting-sla",
        type=float,
        default=180.0,
        help="Waiting SLA threshold in seconds (default: 180).",
    )
    parser.add_argument(
        "--table-vacancy-grace",
        type=float,
        default=2.0,
        help="Seconds before a table is considered vacant (default: 2).",
    )
    args = parser.parse_args()
    if args.waiting_sla <= 0:
        parser.error("--waiting-sla must be positive")
    if args.table_vacancy_grace < 0:
        parser.error("--table-vacancy-grace must be non-negative")
    explicit = [args.tracks, args.events, args.video]
    if args.run:
        run_dir = args.run
    elif any(explicit):
        first = next(path for path in explicit if path is not None)
        run_dir = first.parent
    else:
        run_dir = select_best_run(Path("data/output"))
        print(f"Selected best run: {run_dir}")
    run_server(
        tracks_path=args.tracks or run_dir / "tracks.json",
        events_path=args.events or run_dir / "events.json",
        video_path=args.video or run_dir / "tracked.mp4",
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
        waiting_sla_seconds=args.waiting_sla,
        table_vacancy_grace_seconds=args.table_vacancy_grace,
    )


if __name__ == "__main__":
    main()
