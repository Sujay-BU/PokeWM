"""Training entry point.

    python -m pokewm.train --preset laptop --logdir runs/overnight

The run is resumable: re-invoking with the same --logdir picks up model, optimiser,
frontier archive and counters where they left off. That is the intended way to drive a
multi-day attempt at the full game.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import replace
from pathlib import Path

from .agent.trainer import PokeWorldTrainer, configure_logging
from .config import Config


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config.preset(args.preset)
    if args.logdir:
        cfg.train = replace(cfg.train, logdir=args.logdir)
    if args.envs:
        cfg.train = replace(cfg.train, num_envs=args.envs)
    if args.steps:
        cfg.train = replace(cfg.train, total_steps=args.steps)
    if args.device:
        cfg.train = replace(cfg.train, device=args.device)
    if args.seed is not None:
        cfg.train = replace(cfg.train, seed=args.seed)
    if args.no_llm:
        cfg.llm = replace(cfg.llm, enabled=False)
    if args.llm_model:
        cfg.llm = replace(cfg.llm, model=args.llm_model)
    if args.llm_vision:
        cfg.llm = replace(cfg.llm, use_vision=True)
    if args.no_archive:
        cfg.archive = replace(cfg.archive, enabled=False)
    if args.replay_ratio:
        cfg.train = replace(cfg.train, replay_ratio=args.replay_ratio)
    return cfg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train the Pokemon Red world model")
    ap.add_argument("--preset", default="laptop", choices=["laptop", "cpu", "smoke"])
    ap.add_argument("--logdir", default=None)
    ap.add_argument("--envs", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--replay-ratio", type=float, default=0.0)
    ap.add_argument("--no-llm", action="store_true", help="disable the subgoal proposer")
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--llm-vision", action="store_true",
                    help="use the vision model (slower; see docs/ARCHITECTURE.md)")
    ap.add_argument("--no-archive", action="store_true",
                    help="ablation: disable the Go-Explore frontier archive")
    ap.add_argument("--fresh", action="store_true", help="ignore any existing checkpoint")
    ap.add_argument("--relabel-archive", action="store_true",
                    help="re-score every archived cell against the current milestone "
                         "chain (automatic when the chain changes)")
    ap.add_argument("--reset-policy", action="store_true",
                    help="keep the learned world model but reinitialise actor+critic")
    args = ap.parse_args(argv)

    cfg = build_config(args)
    configure_logging(Path(cfg.train.logdir))
    logging.info("preset=%s logdir=%s", args.preset, cfg.train.logdir)

    trainer = PokeWorldTrainer(cfg, resume=not args.fresh,
                               reset_policy=args.reset_policy,
                               force_relabel=args.relabel_archive)
    trainer.run()
    # Hard-exit rather than unwind.
    #
    # `run()` returns only after its `finally` has written the checkpoint and the
    # archive, so by here every piece of durable state is on disk. What remains is
    # interpreter teardown -- CUDA context, ~29 live threads, the forkserver -- and that
    # was measured hanging for minutes in uninterruptible sleep *after* the final
    # "finished" line. `scripts/train.sh stop` allows 180 s before SIGKILL, so a slow
    # unwind is pure restart latency at best and a killed process at worst.
    logging.shutdown()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
