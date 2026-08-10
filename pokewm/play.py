"""Live GUI viewer: watch the trained policy play in a real emulator window.

Training runs headless for throughput; this is the "GUI based simulator" half. It opens
one SDL2 PyBoy window, loads the latest checkpoint, and drives it with the world model's
actor. It re-reads the checkpoint periodically, so you can leave it running next to a
training job and watch the policy improve.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import torch

from .agent.milestones import (
    NUM_MILESTONES,
    MilestoneTracker,
    achieved_milestone,
)
from .config import Config
from .emulator import maps as M
from .emulator.archive import FrontierArchive
from .emulator.env import NUM_ACTIONS, PokemonRedEnv
from .emulator.ram_map import SYMBOLIC_DIM
from .llm.subgoals import NUM_SUBGOALS
from .wm.actor_critic import ImaginationActorCritic
from .wm.world_model import WorldModel

log = logging.getLogger(__name__)


def load_models(cfg: Config, ckpt_path: Path, device: torch.device):
    channels = cfg.env.frame_stack + cfg.env.seen_map_channels
    wm = WorldModel(
        channels, (cfg.env.frame_h, cfg.env.frame_w), SYMBOLIC_DIM,
        NUM_ACTIONS, NUM_SUBGOALS, cfg.wm,
    ).to(device)
    ac = ImaginationActorCritic(wm.rssm.feat_dim, NUM_ACTIONS, cfg.ac, cfg.wm).to(device)
    steps = 0
    explored: dict | None = None
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        wm.load_state_dict(ck["world_model"])
        ac.load_state_dict(ck["actor_critic"])
        steps = int(ck.get("env_steps", 0))
        # Merge the per-worker novelty snapshots into one. The training workers each
        # hold a slice of the run's coverage; the viewer runs a single env, so it wants
        # the union.
        workers = ck.get("exploration") or []
        if workers:
            explored = {
                "seen_coords": [c for w in workers for c in w.get("seen_coords", [])],
                "seen_maps": sorted({m for w in workers for m in w.get("seen_maps", [])}),
            }
            for key in ("max_event_bits", "max_badges", "max_level_sum",
                        "max_party_size", "max_balls", "max_dex_owned", "max_dex_seen"):
                explored[key] = max((int(w.get(key, 0)) for w in workers), default=0)
        log.info("loaded checkpoint from %d env steps", steps)
    else:
        log.warning("no checkpoint at %s -- watching an untrained policy", ckpt_path)
    wm.eval()
    ac.eval()
    return wm, ac, steps, explored


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Watch the world-model agent play")
    ap.add_argument("--logdir", default=None, help="run directory holding checkpoint.pt")
    ap.add_argument("--preset", default="laptop", choices=["laptop", "cpu", "smoke"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--speed", type=int, default=1, help="0 = unlimited, 1 = real time")
    ap.add_argument("--greedy", action="store_true", help="argmax instead of sampling")
    ap.add_argument("--steps", type=int, default=0, help="0 = run until interrupted")
    ap.add_argument(
        "--from-frontier", action="store_true",
        help="start from the deepest archived save state instead of the game's opening",
    )
    ap.add_argument("--reload-every", type=int, default=20_000,
                    help="re-read the checkpoint every N steps (0 disables)")
    ap.add_argument(
        "--restart-if-stuck", type=int, default=400,
        help="relaunch from a different archived cell after N steps without moving "
             "(0 disables). A mid-training policy can pin itself against a wall or in "
             "a dialogue loop; this keeps the window showing something. Steps spent in "
             "a battle or under script control are exempt -- the overworld position "
             "cannot change there, so they are not evidence of being stuck.",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    cfg = Config.preset(args.preset)
    logdir = Path(args.logdir or cfg.train.logdir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    wm, ac, ckpt_steps, explored = load_models(cfg, logdir / "checkpoint.pt", device)

    env_cfg = replace(cfg.env, render_gui=True, emulation_speed=args.speed,
                      max_episode_steps=10**9)
    env = PokemonRedEnv(env_cfg, cfg.reward, num_subgoals=NUM_SUBGOALS)

    blob = None
    seed_maps: set[int] | None = None
    # Loaded unconditionally, not just under --from-frontier. `--restart-if-stuck`
    # needs a cell to relaunch from, so without this the default invocation had no
    # recovery at all: a wedged policy stayed wedged for the rest of the session, while
    # a training worker in the same state gets a fresh cell within `max_episode_steps`.
    archive = FrontierArchive(cfg.archive)
    archive.load(logdir / "archive")
    if len(archive) == 0:
        archive = None
    if args.from_frontier:
        cell = archive.deepest() if archive else None
        if cell is not None:
            blob = cell.blob
            seed_maps = set(cell.seen_maps)
            # Name the map explicitly. "milestone 5" alone is confusing when the deepest
            # cell happens to sit in a room you have already seen -- the archive stores
            # where the agent *has been*, which is not always where it got furthest.
            log.info(
                "starting from archived cell: %s, milestone %d/%d, %d maps explored "
                "(archive holds %d cells)",
                M.map_name(cell.map_id), cell.milestone, NUM_MILESTONES - 1,
                len(cell.seen_maps), len(archive),
            )
        else:
            log.warning("--from-frontier: archive is empty, starting from the beginning")

    # Seed the novelty memory from the checkpoint. `seen_coords` drives the visited
    # overlay, which is an *observation channel* -- with it empty the policy sees a
    # blank map where every training worker sees a populated one, so the same screen
    # produces a different input here than it did during training.
    if explored:
        env.import_exploration(explored)
        log.info("seeded novelty memory with %d visited tiles across %d maps",
                 len(env.seen_coords), len(env.seen_maps))

    obs, info = env.reset(options={"state_blob": blob} if blob else None)
    tracker = MilestoneTracker()
    if seed_maps:
        # Carry the cell's map history so the readout shows real progress rather than
        # restarting the milestone count at 1 (same fix as the training workers).
        tracker.seen_maps.update(seed_maps)
    tracker.update(info["state"])
    recent: deque[tuple[int, int, int]] = deque(
        maxlen=max(200, args.restart_if_stuck)
    )
    state = wm.rssm.initial(1, device)
    prev_action = torch.zeros(1, NUM_ACTIONS, device=device)
    is_first = torch.ones(1, device=device)

    total = 0
    t0 = time.time()
    try:
        while args.steps == 0 or total < args.steps:
            with torch.no_grad():
                t = {
                    "frame": torch.as_tensor(obs["frame"][None], device=device),
                    "symbolic": torch.as_tensor(obs["symbolic"][None], device=device),
                    "subgoal": torch.as_tensor(obs["subgoal"][None], device=device),
                }
                state = wm.observe_step(state, t, prev_action, is_first)
                idx = ac.policy(state.feature(), greedy=args.greedy)
            action = int(idx.item())
            obs, reward, term, trunc, info = env.step(action)
            newly = tracker.update(info["state"])
            for key in newly:
                log.info("MILESTONE: %s", key)
            prev_action = torch.nn.functional.one_hot(idx, NUM_ACTIONS).float()
            is_first = torch.zeros(1, device=device)
            total += 1

            # A battle or a script is not "stuck": the overworld position cannot change
            # while either owns the screen, so counting those steps made a trainer
            # battle -- which takes hundreds of steps and cannot be fled -- look
            # identical to a sprite pinned against a wall. The viewer then relaunched
            # from a different cell mid-fight, roughly every 400 steps, and the battle
            # was never seen through to its end.
            gs_now = info["state"]
            if gs_now.in_battle == 0 and gs_now.joy_ignore == 0:
                recent.append(gs_now.position)
            else:
                recent.clear()
            if total % 250 == 0:
                gs = info["state"]
                # `moved` is the number of distinct tiles in the last 200 steps. Without
                # it a policy pinned to a single tile is indistinguishable from a frozen
                # emulator in the log -- which is exactly how this looked when the policy
                # had collapsed into mashing A at one spot.
                moved = len(set(recent))
                log.info(
                    "%6d steps | %5.0f sps | %-20s (%3d,%3d) | milestone %2d/%d | "
                    "badges %d | events %3d | %2d tiles/200 steps%s",
                    total, total / max(time.time() - t0, 1e-9), gs.map_name, gs.x, gs.y,
                    tracker.index, NUM_MILESTONES - 1, gs.badge_count,
                    gs.event_flag_bits, moved,
                    "  <-- stuck" if moved <= 2 else "",
                )
            # A mid-training policy can wedge itself against a wall or in a dialogue
            # loop. Rather than let the window sit on a motionless sprite, jump to a
            # different archived cell -- logged, so it is never mistaken for progress.
            if (
                args.restart_if_stuck
                and archive is not None
                and len(recent) >= args.restart_if_stuck
                and len({p for p in list(recent)[-args.restart_if_stuck:]}) <= 2
            ):
                nxt = archive.sample() or archive.deepest()
                if nxt is not None:
                    log.info(
                        "stuck at %s for %d steps; relaunching from another cell (%s, "
                        "milestone %d)",
                        info["state"].map_name, args.restart_if_stuck,
                        M.map_name(nxt.map_id), nxt.milestone,
                    )
                    blob, seed_maps = nxt.blob, set(nxt.seen_maps)
                    obs, info = env.reset(options={"state_blob": blob})
                    tracker = MilestoneTracker()
                    tracker.seen_maps.update(seed_maps)
                    tracker.update(info["state"])
                    state = wm.rssm.initial(1, device)
                    is_first = torch.ones(1, device=device)
                    recent.clear()

            if args.reload_every and total % args.reload_every == 0:
                wm, ac, new_steps, _ = load_models(
                    cfg, logdir / "checkpoint.pt", device)
                if new_steps != ckpt_steps:
                    ckpt_steps = new_steps
                    state = wm.rssm.initial(1, device)
            if term:
                obs, info = env.reset(options={"state_blob": blob} if blob else None)
                state = wm.rssm.initial(1, device)
                is_first = torch.ones(1, device=device)
    except KeyboardInterrupt:
        pass
    finally:
        gs = env.state
        if gs is not None:
            log.info(
                "stopped at %s, milestone %d/%d (%s), %d badges",
                gs.map_name, tracker.index, NUM_MILESTONES - 1,
                (achieved_milestone(tracker.index).key
                 if achieved_milestone(tracker.index) else "none"),
                gs.badge_count,
            )
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
