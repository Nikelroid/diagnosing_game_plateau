"""Run the survey and write the data files every table and figure is built from.

    python scripts/run_survey.py                 # both engines
    python scripts/run_survey.py --engine rlcard # just the fast one
    python scripts/run_survey.py --games 20      # quick pass while developing

A game that fails is recorded as a row with an error rather than aborting the run, because the
list is long and a single unloadable game should not cost the other sixty.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from hidden_info_survey.catalog import OPENSPIEL_GAMES, RLCARD_GAMES  # noqa: E402


def run(specs, measure, out_path, **kwargs):
    rows, started = [], time.time()
    for spec in specs:
        try:
            row = measure(spec, **kwargs)
            rows.append(row)
            print(f"  ok   {spec.label:38s} {row['seconds']:6.1f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "game": spec.key,
                    "label": spec.label,
                    "family": spec.family,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"  FAIL {spec.label:38s} {type(exc).__name__}", flush=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"rows": rows, "wall_seconds": round(time.time() - started, 1)}, f, indent=2)
    failed = sum(1 for r in rows if "error" in r)
    print(f"wrote {out_path}  ({len(rows) - failed} measured, {failed} failed)\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["openspiel", "rlcard", "both"], default="both")
    ap.add_argument("--games", type=int, default=100, help="random playouts per game")
    ap.add_argument("--nodes", type=int, default=40_000, help="node cap for exact enumeration")
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    args = ap.parse_args()

    if args.engine in ("openspiel", "both"):
        from hidden_info_survey.openspiel_probe import measure

        print(f"OpenSpiel: {len(OPENSPIEL_GAMES)} games")
        run(
            OPENSPIEL_GAMES,
            measure,
            os.path.join(args.data_dir, "openspiel.json"),
            max_nodes=args.nodes,
            n_games=args.games,
        )

    if args.engine in ("rlcard", "both"):
        from hidden_info_survey.rlcard_probe import measure

        print(f"RLCard: {len(RLCARD_GAMES)} games")
        run(
            RLCARD_GAMES,
            measure,
            os.path.join(args.data_dir, "rlcard.json"),
            n_games=min(args.games, 60),
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
