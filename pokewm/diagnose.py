"""On-demand diagnosis of a run: is it progressing, and if not, why?

    python -m pokewm.diagnose --logdir runs/overnight

Encodes the checks that actually found every stall in this project, so they can be run
in seconds instead of rediscovered each time:

1. Progress signals over time -- milestone, archive growth, coverage, story flags.
2. Known failure signatures -- entropy at the ln|A| ceiling or collapsed, negative
   imagined reward, a starved frontier, a frozen archive.
3. Where the archive actually sends the agent, versus where the next milestone is.
4. **An active reachability probe.** Metrics cannot separate "the agent cannot do this"
   from "this cannot be measured" or "this is gated". Driving the frontier states with a
   dumb high-entropy policy can, and that distinction is the one worth having: a
   milestone the agent was completing in ~25 steps once looked permanently unreachable
   purely because of a two-byte RAM offset.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
from pathlib import Path

from .agent.milestones import NUM_MILESTONES, achieved_milestone, next_milestone
from .agent.stall import ReachabilityProbe, StallDetector
from .config import Config
from .emulator import maps as M
from .emulator.archive import FrontierArchive
from .emulator.env import NUM_ACTIONS

log = logging.getLogger(__name__)


def _load_metrics(logdir: Path) -> list[dict]:
    path = logdir / "metrics.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def report_progress(rows: list[dict], window: int) -> StallDetector:
    det = StallDetector(
        window_steps=window,
        hard_window_steps=Config.preset("laptop").train.stall_hard_window,
        action_space=NUM_ACTIONS,
    )
    if not rows:
        print("  no metrics yet")
        return det
    # Metrics are appended across restarts, so counters can jump backwards. Only the
    # current process's tail is meaningful for a stall verdict.
    start = 0
    for i in range(1, len(rows)):
        if rows[i].get("run/env_steps", 0) < rows[i - 1].get("run/env_steps", 0):
            start = i
    tail = rows[start:]
    print(f"  {len(tail)} samples since the last restart")
    print(f"  {'env_steps':>12} {'cells':>7} {'coords':>9} {'maps':>6} {'events':>7} "
          f"{'H':>6} {'imagR':>9}")
    for r in tail[:: max(1, len(tail) // 8)]:
        print(f"  {r.get('run/env_steps',0):>12,.0f} {r.get('archive/cells',0):>7.0f} "
              f"{r.get('progress/unique_coords',0):>9.1f} "
              f"{r.get('progress/unique_maps',0):>6.2f} "
              f"{r.get('progress/events',0):>7.2f} {r.get('policy/entropy',0):>6.3f} "
              f"{r.get('policy/imag_reward',0):>+9.4f}")
    for r in tail:
        det.update(int(r.get("run/env_steps", 0)), {
            "milestone": r.get("run/best_milestone", 0.0),
            "archive_cells": r.get("archive/cells", 0.0),
            "unique_coords": r.get("progress/unique_coords", 0.0),
            "max_events": r.get("progress/events", 0.0),
            "entropy": r.get("policy/entropy", 0.0),
            "imag_reward": r.get("policy/imag_reward", 0.0),
            "frontier_frac": r.get("archive/frontier_frac", 1.0),
        })
    return det


def report_archive(cfg: Config, logdir: Path, best_milestone: int) -> FrontierArchive:
    arch = FrontierArchive(cfg.archive, seed=0)
    arch.load(logdir / "archive")
    if len(arch) == 0:
        print("  archive empty")
        return arch
    per_ms = collections.Counter(c.milestone for c in arch._cells.values())
    per_map = collections.Counter(M.map_name(c.map_id) for c in arch._cells.values())
    print(f"  {len(arch)} cells, max milestone {arch.max_milestone}")
    print(f"  per milestone: {dict(sorted(per_ms.items()))}")
    print(f"  top maps     : {dict(per_map.most_common(5))}")

    nxt = next_milestone(best_milestone)
    if nxt is not None:
        requested = nxt.targets()
        arch.set_target_maps(requested)
        # Score against the *effective* target, not the requested one. A "reach map X"
        # milestone names a map with no cells, and the archive falls back to the deepest
        # map it has actually reached. Reporting against the request made a working
        # fallback read as 0.0% on-target -- a detector that cries wolf gets ignored.
        effective = arch._target_maps
        picks = collections.Counter()
        ms_picks = collections.Counter()
        for _ in range(2000):
            c = arch.sample()
            if c:
                picks[M.map_name(c.map_id)] += 1
                ms_picks[c.milestone] += 1
        total = max(sum(picks.values()), 1)
        req_txt = ", ".join(sorted(M.map_name(t) for t in requested)) or "(none)"
        print(f"  next milestone: {nxt.key} -> targets [{req_txt}]")
        if effective != requested:
            eff_txt = ", ".join(sorted(M.map_name(t) for t in effective)) or "(none)"
            print(f"  target unreached; falling back to deepest reached: [{eff_txt}]")
        print("  restores land on:")
        for k, v in picks.most_common(5):
            print(f"      {k:<24}{100*v/total:6.1f}%")
        eff_names = {M.map_name(t) for t in effective}
        frac = sum(v for k, v in picks.items() if k in eff_names) / total
        verdict = ("OK" if frac > 0.5 else
                   "LOW -- the archive is not sending the agent to the objective")
        print(f"  on-target restores: {frac:.1%}  [{verdict}]")

        # Landing on the right *map* is not enough. Milestone level encodes irreversible
        # world state -- gates opened, key items held -- so a restore one level short can
        # put the agent on the correct map in a world where the objective is sealed off.
        # Exactly that hid a 2.0M-step stall behind a 99.8% on-target reading: the map
        # was right and the Viridian gate was shut in 80% of those launches.
        print("  restores by milestone level:")
        for lvl, v in sorted(ms_picks.items(), reverse=True)[:4]:
            print(f"      milestone {lvl:<3}{100*v/total:6.1f}%")
        front = sum(v for k, v in ms_picks.items() if k >= arch.max_milestone) / total
        fverdict = ("OK" if front > 0.5 else
                    "STALE -- most restores predate the current milestone, so the "
                    "next one may be unreachable from them")
        print(f"  frontier-level restores: {front:.1%}  [{fverdict}]")
    return arch


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Diagnose a training run")
    ap.add_argument("--logdir", default=None)
    ap.add_argument("--preset", default="laptop", choices=["laptop", "cpu", "smoke"])
    ap.add_argument("--window", type=int, default=750_000,
                    help="env steps considered for the stall verdict")
    ap.add_argument("--probe", action="store_true",
                    help="actively test whether the next milestone is reachable "
                         "(boots an emulator; a few seconds per cell)")
    ap.add_argument("--probe-cells", type=int, default=6)
    ap.add_argument("--probe-steps", type=int, default=4000)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    cfg = Config.preset(args.preset)
    logdir = Path(args.logdir or cfg.train.logdir)

    rows = _load_metrics(logdir)
    best = int(rows[-1].get("run/best_milestone", 0)) if rows else 0
    done = achieved_milestone(best)
    nxt = next_milestone(best)
    print(f"\n=== run: {logdir} ===")
    print(f"  milestone {best}/{NUM_MILESTONES - 1}")
    print(f"  achieved : {done.label if done else '(nothing yet)'}")
    print(f"  next     : {nxt.label if nxt else 'GAME COMPLETE'}")

    print("\n=== progress ===")
    det = report_progress(rows, args.window)
    report = det.check()
    print("\n=== verdict ===")
    print("  " + report.describe().replace("\n", "\n  "))

    print("\n=== archive ===")
    arch = report_archive(cfg, logdir, best)

    if args.probe:
        print("\n=== active reachability probe ===")
        probe = ReachabilityProbe(cfg, num_cells=args.probe_cells,
                                  steps_per_cell=args.probe_steps)
        result = probe.run(arch, nxt)
        print("  " + result.describe().replace(". ", ".\n  "))

    print()
    return 0 if not report.stalled else 1


if __name__ == "__main__":
    raise SystemExit(main())
