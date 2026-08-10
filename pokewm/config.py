"""Central configuration.

Every tunable lives here as a frozen-ish dataclass so that a run is fully described by
one serialisable object, which is written into each checkpoint. `Config.preset()` provides
three profiles: `laptop` (the 6 GB / 16 GB GPU profile this repository was developed and
validated on), `cpu` (a ~4M-parameter fallback for machines without a usable GPU), and
`smoke` (seconds-long, for tests).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROM = REPO_ROOT / "roms" / "Pokemon - Red Version (USA, Europe) (SGB Enhanced).gb"

# Verified good dump. The env refuses to start on a mismatch unless explicitly waived,
# because every RAM offset in `ram_map` is only valid for this revision.
ROM_SHA1 = "ea9bcae617fdf159b045185467ae58b2e4a48b9a"


@dataclass
class EnvConfig:
    rom_path: str = str(DEFAULT_ROM)
    check_rom_hash: bool = True

    # Screen is 160x144. We downsample by 2 and keep luminance only.
    frame_h: int = 72
    frame_w: int = 80
    frame_stack: int = 4

    # Emulator ticks executed per agent action. 24 ticks ~= 0.4 s of game time, which is
    # roughly one overworld step or one text advance.
    action_frames: int = 24
    # Movement in the overworld needs a face-then-step; PyBoy button holds are ticks.
    button_hold_frames: int = 8

    max_episode_steps: int = 8192
    # Extra steps an episode may run past its budget to finish a battle it is in.
    #
    # Every episode ends by truncation, and landing mid-fight abandoned the battle: the
    # worker restored from a different archive cell and the outcome never happened. That
    # matters most for trainer battles, which cannot be fled -- an abandoned one is a
    # fight the agent never has to resolve, so menu-cycling until the clock runs out was
    # a way to avoid fighting at all.
    #
    # Bounded so a genuinely stuck battle cannot hold an episode open indefinitely.
    battle_grace_steps: int = 512
    # Whether a party wipe ends the episode.
    #
    # Off, and this is a correctness fix rather than a preference. In Pokemon Red a
    # blackout is not an ending: the player is teleported to the last Pokecenter, loses
    # some money, and play continues. Modelling it as a terminal state handed the agent
    # an escape hatch from any negative per-step reward -- terminating cost it -10 while
    # running the episode out cost -91, so it began blacking out deliberately. Treating
    # it as the setback it actually is removes the incentive structurally, rather than
    # relying on the penalty being tuned large enough to outweigh the horizon.
    terminate_on_wipe: bool = False
    # Hard cap on emulator speed. 0 = unbounded (headless training).
    emulation_speed: int = 0
    render_gui: bool = False

    # Exploration bookkeeping
    coord_bucket: int = 1  # tiles per novelty bucket
    seen_map_channels: int = 1  # binary "visited" overlay appended to the frame stack
    # Length of the short-term position memory used by the anti-dither penalty. Long
    # enough to catch a left/right oscillation, short enough that legitimately walking
    # back down a corridor costs almost nothing.
    dither_window: int = 32

    save_state_dir: str = str(REPO_ROOT / "runs" / "states")
    init_state: str | None = None  # optional .state to always boot from


@dataclass
class RewardConfig:
    """Weights for the shaped return.

    The *only* terms that touch irreversible game progress (events, badges, dex) are
    given first-visit-only credit, which keeps the shaping potential-like and therefore
    policy-invariant in the limit (see docs/PROOF.md §4).
    """

    event: float = 4.0  # per newly-set story flag
    badge: float = 64.0  # per gym badge
    # Per party level gained. Raised 0.6 -> 2.0 because levels are the game's mandatory
    # progression currency and were priced as an afterthought.
    #
    # Measured over 18M steps: `level_sum` never left 3.75-4.5 -- one starter that never
    # levelled once. The reason is an incentive, not an inability. Winning a wild battle
    # pays a fraction of a level (0.6 x that fraction); losing pays `faint`, then at -10
    # a wipe. Engaging was worth it only above ~97% win probability, so a level-5 party
    # correctly learned to avoid every fight -- and a party that never fights never
    # levels, which caps the run permanently well before Brock's level-14 Onix.
    #
    # At 2.0, grinding the ~8 levels Brock needs pays +16: real, and still far under a
    # badge at 64.
    level: float = 2.0
    dex_owned: float = 2.0
    # Per party member, on a monotone maximum. A second Pokemon is a *life*.
    #
    # Nothing rewarded party size at all: `dex_owned` pays 2.0 for a new species, so
    # catching a duplicate added a life and paid zero. The agent reached 44M env steps
    # having never caught anything -- party size 1 in every archived cell -- which makes
    # a single faint a blackout and ends the run. That is why 72 of 90 wild encounters
    # in Viridian Forest ended in a wipe.
    #
    # 8.0 is deliberately above a level (2.0) and a species (2.0): a fresh level-3 catch
    # is worth more than three levels on the one Pokemon it already has, because it is
    # the difference between losing a fight and losing the run. Still far under a badge.
    #
    # Monotone: paid only above the previous maximum, so depositing and withdrawing at a
    # PC cannot farm it.
    party_member: float = 8.0
    # Paid when a battle ends with the opponent fainted and the party still standing.
    #
    # Nothing rewarded *winning* a fight. `enemy_damage` pays at most 1.0 in total for
    # taking the opponent from full to zero, while `faint` costs 5.0, and fleeing pays
    # exactly 0 at no risk -- so fleeing strictly dominated fighting and the policy
    # learned to flee. Measured with the live policy from healthy archive restores: 40
    # battles in 4000 steps, party level sum 8 -> 8, not one level gained, with an
    # in-battle action mix of 23% B (cancel) against 15% A. `reward/level` had never
    # fired in the whole run.
    #
    # 3.0 makes fighting positive-EV once the win probability passes roughly 0.5
    # (3.0*p + ~1.0 damage - 5.0*(1-p) > 0), which is the right threshold: fight what
    # you can beat, flee what you cannot. It is not farmable beyond what is intended --
    # repeatedly winning wild battles *is* levelling, which is the behaviour the run
    # needs, and it is rate-limited by party HP and by the Pokecenter round trip.
    battle_won: float = 3.0
    # Balls carried, as a *potential* -- not a monotone maximum.
    #
    # `party_member` pays 8.0 for a catch the agent had no way to make: every one of 205
    # archived Viridian Forest cells carried {Town Map, Potion} and zero balls, so the
    # CATCH subgoal was unsatisfiable. Buying is the precondition, and nothing paid for it.
    #
    # The first version of this term was monotone, which is wrong for anything spendable.
    # Oak hands over 5 balls (EVENT_GOT_POKEBALLS_FROM_OAK was set in 405 of 607 live
    # cells) and the agent threw them all; `max_balls` was then stuck at a value the bag
    # could never exceed again. Measured at 83.5M env steps it sat at 1 while all 607
    # cells carried 0 balls, so buying a ball would have paid exactly nothing. Monotone
    # is right for levels and party size, which cannot decrease; it is wrong here.
    #
    # Symmetric, so it telescopes and cannot be farmed: throwing costs precisely what
    # buying paid. A wasted throw costing 1.5 is the intended lesson -- weaken it first.
    # Catching still nets +8.5 (`party_member` 8.0 plus `dex_owned` 2.0, less the ball),
    # so the catch stays far more valuable than the hoard.
    ball: float = 1.5
    ball_cap: int = 6
    ball_price: int = 200  # Poke Mart price; gates the BUY_ITEMS subgoal on affordability
    dex_seen: float = 0.2
    new_map: float = 3.0  # first time entering a map
    # Paid on transitions between on-path maps, proportional to the change in
    # `map_rank`. The only *directed* term in this function.
    #
    # Everything else is first-visit-only, so on already-covered ground the agent gets
    # nothing at all: new_tile and new_map are spent, and the epistemic bonus decays to
    # zero precisely where the world model fits well. Restored onto Route 1 with Viridian
    # north and Pallet south, the agent therefore kept re-running the delivery route it
    # had spent millions of steps learning and walked *south* -- 38 frontier cells
    # accumulated in Pallet Town against 2 on Route 1.
    #
    # Telescoping (paid on transitions, not per step) makes any round trip worth exactly
    # zero, so it cannot be farmed by oscillating between two maps, and it adds no
    # standing per-step cost -- which is what makes it safe here, given that a -0.002
    # per-step term once made blacking out the optimal policy.
    #
    # Raised 0.15 -> 0.5 on measurement. At 0.15 it was simply outbid. Novelty-based
    # exploration goes where the unexplored *volume* is, and there is far more of that
    # behind the frontier than ahead of it: a Route 2 -> Route 1 excursion opens ~40 new
    # tiles worth 0.8 against a 0.6 charge, so retreating paid. The mean payout per
    # on-path transition fell monotonically -0.024 -> -0.122 over 500k steps -- the agent
    # was actively learning to walk backwards while the archive kept restoring it to the
    # frontier, which is an expensive way to stand still.
    #
    # Still an order of magnitude under the 4.0 an event flag pays, so the 2.5 now
    # charged for the Viridian -> Pallet leg of the parcel run stays far cheaper than
    # completing it. Telescoping keeps this safe to raise: it is charged on transitions,
    # never per step, so a bigger weight cannot create a standing cost of the kind that
    # once made blacking out optimal.
    map_progress: float = 0.5
    new_tile: float = 0.02  # first time standing on a (map, x, y)
    # Penalty for stepping onto a tile already occupied within the last
    # `dither_window` steps. Targets the degenerate left/right oscillation the policy
    # actually converged to in the first long run (measured: 74% of probability mass on
    # two opposing actions, 13 distinct positions in 600 steps). Sized to cancel a
    # `new_tile` bonus so dithering nets out negative while genuine exploration pays.
    # PokeRL (2026) reports loop episodes falling 41.2% -> 4.7% from an equivalent term;
    # the original design here omitted it on the assumption that the epistemic bonus
    # would suffice, which this run disproved.
    #
    # Sized well below the typical epistemic bonus (~+0.02) so that the per-step baseline
    # stays positive while exploring. At -0.02 it dominated everything: mean imagined
    # reward settled at -0.0133/step, making a full 6851-step episode worth -91 and
    # turning a -2.0 blackout into a +89 shortcut. Episode termination rose 0.23 -> 0.41
    # as the agent learned to kill itself. A per-step penalty must never exceed what the
    # agent can earn per step, or ending the episode becomes the optimal policy.
    dither: float = -0.005
    # Party HP as a *potential*: paid on every change, both directions.
    #
    # Replaces the old one-sided `heal` bonus, which paid for recovering HP but charged
    # nothing for losing it -- a damage/heal cycle was free money, and the only reason it
    # was never farmed is that the agent never healed at all. Symmetric means a
    # damage-then-heal round trip telescopes to exactly zero, so nothing is farmable,
    # while sitting at low HP is strictly worse than sitting at full HP. That standing
    # gap is what makes a Pokecenter trip worth taking.
    #
    # Archived frontier states were measured at party_hp_frac 0.40 with a single
    # Pokemon -- the agent had been fighting at 40% health indefinitely.
    #
    # 3.0 means healing 0.4 -> 1.0 pays 1.8. A Pokecenter is off-path (map_rank -1) so
    # entering is free, and the walk there and back telescopes to zero under
    # `map_progress`, leaving only the time cost to beat.
    hp_potential: float = 3.0
    # Fraction of the opponent's max HP removed, per step.
    #
    # The one dense signal for actually fighting. Everything else about combat is sparse
    # (a level, a badge), so "attack" and "open the bag again" looked equally good, and
    # the agent spent trainer battles cycling menus. Winning a full-health opponent pays
    # 1.0 on top of the level reward.
    enemy_damage: float = 1.0
    # Per-step cost once a *trainer* battle has gone `battle_stall_grace` steps with
    # neither side's HP changing.
    #
    # Trainer battles cannot be fled -- selecting RUN prints a refusal and burns the turn
    # -- so an agent that keeps choosing it stalls forever with no other term objecting.
    # Deliberately menu-agnostic: it targets "this battle is going nowhere" rather than a
    # hardcoded RUN cursor index, which is ROM-revision-specific and which measurement
    # here showed to be nested and unstable (wMaxMenuItem shifts 7 -> 1 -> 3 as submenus
    # open).
    #
    # Sized an order of magnitude below `enemy_damage` so attacking always dominates
    # waiting, and it can never become an incentive to end the episode: it only applies
    # inside a trainer battle, which the agent can leave by winning.
    battle_stall: float = -0.02
    battle_stall_grace: int = 24
    # Blacking out is a setback, not an ending -- see EnvConfig.terminate_on_wipe.
    #
    # Reduced -10 -> -4. The penalty has to be read against what a level is worth, since
    # together they set the price of entering a battle at all. At -10 against level 0.6 a
    # wipe cost ~17 levels of progress, which made combat unplayable arithmetic; at -5
    # against 2.0 it costs 2.5, which is about what a real blackout costs in walk-back
    # time. Kept above `event` so a wipe still outweighs a story flag, which is a
    # separate invariant the test suite pins. The agent still loses its position -- that is the setback, and it is a real
    # one -- but the teleport itself is no longer *also* billed as backward travel
    # (see `_blackout_pending`).
    faint: float = -5.0
    money: float = 0.0002
    # Kept an order of magnitude below the intrinsic bonus. At the original -0.002 it
    # exactly cancelled the measured epistemic drive (0.0185 JSD x 0.10 = 0.00185/step),
    # leaving imagined return slightly *negative*: the agent's best imagined future was
    # to do nothing, and exploration stalled for 2.7M steps.
    step_cost: float = -0.0005
    # Intrinsic terms
    # World-model ensemble disagreement, applied in imagination. JSD is bounded by
    # ln(ensemble_size) = 1.386, so even at weight 1.0 this cannot exceed ~1.4 per step
    # against a badge worth 64 -- bounded, and self-extinguishing as the model fits.
    epistemic: float = 1.0
    # Whether to ALSO add the disagreement bonus to the extrinsic reward stored in
    # replay. Default off, for two reasons:
    #   1. It would be double counted -- the actor-critic already adds the bonus to
    #      imagined rewards, and a stored bonus is additionally baked into whatever the
    #      reward head learns to predict.
    #   2. The bonus decays as the model fits, so storing it makes the reward head's
    #      regression target nonstationary and learned from stale data.
    # Plan2Explore and Simulus both apply intrinsic reward in imagination only.
    env_epistemic: bool = False
    subgoal: float = 3.0  # LLM-proposed subgoal satisfied
    # Party HP below which HEAL is forced as the active subgoal, overriding the LLM.
    #
    # The proposer is single-flight with a 30 s cooldown, round-robin over 8 workers, so
    # a worker's suggestion is minutes old -- far too slow for "you are at 10% HP", which
    # is the state the agent lived in: every archived Viridian Forest cell held one
    # level-6 Pokemon at 10-40% health, losing 72 of 90 wild encounters.
    #
    # 0.5 rather than higher: healing is worth a detour when half the party's health is
    # gone, not for a scratch, and the walk to a Pokecenter costs real steps.
    # Bonus for reaching a Pokecenter while below `heal_subgoal_hp`.
    #
    # `hp_potential` is symmetric, so healing refunds the damage and nothing more: there
    # is no net reason to make the trip, only to avoid having needed it. This pays for
    # the journey. Paid once per bout of damage, re-armed only when HP falls below the
    # threshold again, so pacing in and out of the building earns nothing.
    #
    # 2.0 covers a few hundred steps of walking at `new_tile` rates while staying under
    # an event flag at 4.0.
    # Curing a status ailment, as a symmetric potential: contracting one costs exactly
    # what curing it pays, so deliberately getting poisoned is not a way to earn.
    #
    # Status was read from RAM from the start but never surfaced in the observation, so
    # the agent could not see that it was poisoned -- and poison costs HP every few steps
    # in the overworld. A poisoned one-Pokemon party bleeds out while walking, which is a
    # loss the agent had no way to anticipate or attribute.
    status_potential: float = 3.0
    heal_visit: float = 2.0
    heal_subgoal_hp: float = 0.5
    # Force CATCH_POKEMON while in a wild battle and the party is smaller than this.
    #
    # The opportunity to catch exists only during a wild encounter, which lasts a few
    # seconds -- far inside the proposer's minutes-long latency. Without a reactive cue
    # the agent has never once taken it.
    #
    # 3 rather than 6: the aim is to stop one faint from ending the run, not to fill the
    # party. Beyond three the detour stops paying for itself.
    catch_subgoal_party: int = 3
    # Clamp on total per-step reward before symlog, guards against RAM glitches.
    clip: float = 128.0


@dataclass
class WorldModelConfig:
    """RSSM sized for 6 GB VRAM.

    Discrete-latent RSSM (Hafner et al., DreamerV3) with the token-model era additions
    that survived replication: symlog encoding, free-bits KL balancing, a two-hot
    (HL-Gauss) return head, and an observation-head ensemble for epistemic uncertainty
    (Simulus, 2025).
    """

    deter: int = 1024  # GRU recurrent state size
    stoch: int = 32  # number of categorical variables
    classes: int = 32  # classes per categorical
    hidden: int = 512  # MLP width
    cnn_depth: int = 40  # base channel count of the conv encoder
    layers: int = 3

    kl_free: float = 1.0  # free nats
    kl_dyn_scale: float = 0.5
    kl_rep_scale: float = 0.1

    ensemble_size: int = 4  # observation-prediction heads for disagreement
    horizon: int = 15  # imagination rollout length
    # 32x32 rather than DreamerV3's 16x64: identical replay volume per update, but half
    # the sequential depth in the RSSM filtering loop. On a laptop GPU that loop is
    # kernel-launch bound, and the wider/shallower geometry measured 2.30 vs 1.85
    # updates/s at the same VRAM (see docs/ARCHITECTURE.md "Throughput").
    batch_size: int = 32
    batch_length: int = 32
    # Imagination starts are subsampled from the batch_size*batch_length posterior
    # states. All 1024 costs ~30% more time for no measured benefit at this scale.
    imag_batch: int = 256

    lr: float = 1e-4
    eps: float = 1e-8
    grad_clip: float = 1000.0
    weight_decay: float = 0.0

    # Return head discretisation
    bins: int = 255
    bin_low: float = -20.0
    bin_high: float = 20.0

    subgoal_dim: int = 32  # width of the subgoal embedding conditioning the actor


@dataclass
class ActorCriticConfig:
    hidden: int = 512
    layers: int = 3
    lr: float = 3e-5
    eps: float = 1e-5
    grad_clip: float = 100.0
    gamma: float = 0.997
    lam: float = 0.95
    # DreamerV3 uses 3e-4; at that value the policy collapsed onto two opposing actions
    # (74% combined) and stopped pressing A almost entirely (1.1%). Raising it to 3e-3
    # over-corrected in the other direction: with early advantages of only ~0.006, the
    # entropy term (3e-3 x 1.95 = 0.0058) dominated the policy gradient and the actor sat
    # at *exactly* ln(7) -- uniform random -- for 400k steps before the critic built
    # enough signal to escape.
    #
    # 1e-3 then went too far the other way: entropy fell to 0.24 (of a 1.946 maximum).
    # 2e-3 collapsed again. Four hand-picked constants produced two collapses and one
    # pinned-at-uniform, because the right coefficient depends on the advantage scale,
    # which changes by two orders of magnitude over a run (adv_std 0.006 -> 0.3).
    #
    # So it is no longer hand-picked: `entropy` is only the initial value, and the
    # coefficient is tuned online to hold a target entropy (see `entropy_adaptive`).
    entropy: float = 2e-3
    # Automatic entropy tuning, as in SAC (Haarnoja et al. 2018): treat the coefficient
    # as a dual variable and adjust it so realised entropy tracks a target. This removes
    # an entire recurring class of stall -- a collapsed policy cannot stumble into the
    # scripted "face an NPC and press A" interactions the game is gated on, and a policy
    # pinned at uniform never commits to anything.
    entropy_adaptive: bool = True
    # Target as a fraction of ln|A|. 0.45 keeps the policy clearly committed while
    # leaving enough spread to discover single-button interactions.
    entropy_target_frac: float = 0.45
    # Step size on log(alpha). At 3e-4 the coefficient needed ~9000 updates (~2 h at the
    # measured 1.2 upd/s) to cross its own band, which is slower than the failure it is
    # supposed to correct.
    entropy_lr: float = 1e-3
    entropy_min: float = 1e-5
    # Ceiling. A hand-set 3e-3 was already enough to pin the policy at exactly ln|A|
    # earlier in this run, so 1e-2 is generous headroom; the point of the bound is to
    # cap how far the dual variable can overshoot before the gap flips sign, since every
    # unit of overshoot costs an equal stretch of recovery on the way back down.
    entropy_max: float = 1e-2
    # Return normalisation percentile range (DreamerV3 §"robust returns")
    return_norm_low: float = 5.0
    return_norm_high: float = 95.0
    return_norm_decay: float = 0.99
    return_norm_limit: float = 1.0
    slow_critic_update: int = 1
    slow_critic_fraction: float = 0.02


@dataclass
class ReplayConfig:
    # Transitions held in RAM. Frames are stored unstacked (see wm/replay.py), so the
    # cost is ~11.6 KB/step => ~2.3 GB at this capacity, which fits alongside 8 emulator
    # processes and the CUDA host allocations on a 16 GB machine.
    capacity: int = 200_000
    min_size: int = 4_000
    # Simulus-style prioritised world-model replay: alpha of each batch is drawn by
    # softmax over per-sequence model loss, the rest uniform.
    prioritized_fraction: float = 0.5
    priority_temperature: float = 1.0
    directory: str = str(REPO_ROOT / "runs" / "replay")


@dataclass
class ArchiveConfig:
    """Go-Explore style frontier archive of emulator save states.

    This is the component that makes a 400-hour game tractable: it converts one
    astronomically long-horizon problem into a chain of short-horizon ones by allowing
    resets directly onto the current progress frontier.
    """

    enabled: bool = True
    max_cells: int = 4_000
    # Cap on cells stored per milestone level.
    #
    # docs/PROOF.md §3.2 bounds the frontier-selection probability by
    # sigma >= (n_0 / n_max) * 0.233, where n_0 is the number of frontier cells and
    # n_max the largest per-level population. The proof assumed that ratio is order 1.
    # In practice shallow levels accumulate without bound -- eviction only triggers at
    # `max_cells`, so a real run reached 83 cells in Pallet Town against 11 on Route 1,
    # giving n_0/n_max = 0.13 and sigma ~ 0.03. Only 3% of restores landed on the
    # frontier and progress crawled.
    #
    # Capping each level bounds n_max directly and lifts sigma to ~0.8 at these
    # populations, which is the regime the proof's expected-time bound assumes.
    max_cells_per_level: int = 32
    # Edge, in tiles, of the coarse position bucket that forms part of a cell key.
    #
    # Keying cells on progress flags alone froze the archive for 2M steps in the first
    # long run: everything between "on Route 1" and "reached Viridian City" sets no
    # story flag and crosses no milestone, so the key never changed, no cells were
    # added, and Go-Explore was effectively disabled inside every long phase. Position
    # is what Go-Explore actually keys on (a downsampled state); bucketing at 8 tiles
    # keeps the count manageable -- roughly 25 buckets for a large map -- while letting
    # the frontier advance across a route.
    #
    # Refined 8 -> 4 on measurement. The bucket edge is the *ratchet step*: within one
    # bucket the archive cannot tell "just arrived" from "at the far edge", so every
    # restore may hand back ground the agent had already crossed.
    #
    # Route 2 made this concrete. Its accessible southern section is 10 tiles wide by 24
    # tall, giving six cells at edge 8; the archive held five and stopped growing --
    # saturated, not stalled. But bucket row 6 spans y 48-55, and the entrance to
    # Viridian Forest is a single ungated warp tile at (3, 43). A cell in that row can
    # hold a state saved at y=55, exactly where the agent arrives from Viridian, so a
    # restore could cost it the entire northward walk it had already made. The run held
    # milestone 10 for 1.3M steps standing at that doorstep.
    #
    # Halving the edge splits that row into y 48-51 and 52-55 and adds a rung at 44-47,
    # so partial progress up the corridor is banked instead of discarded. Cost is 4x
    # cells on the frontier level (superseded levels stay capped at
    # `max_cells_per_level`) at 164 KB per save state -- ~110 MB rather than ~50 MB.
    position_bucket: int = 4
    # Probability an episode starts from an archived frontier cell rather than the ROM
    # boot state. Annealed up as the frontier deepens.
    restore_prob: float = 0.85
    # Softmax temperature over cell scores when sampling which cell to restore.
    temperature: float = 1.0
    # Cells whose visit count exceeds this get down-weighted (count-based bonus).
    novelty_weight: float = 1.0
    # Weight on how far along the critical path a cell's *map* is.
    #
    # Milestone alone is a history counter: a state that reached Route 1 and wandered
    # back into Red's House still scores milestone 6. Measured on a live run, 50 of the
    # 62 frontier-level cells were backtracked into the starting area, so restores
    # mostly landed behind the frontier and Route 1 coverage never advanced past its
    # four southernmost position buckets in 217k steps. Map rank is a property of the
    # state, not of the trajectory, and separates "has been far" from "is far".
    map_rank_weight: float = 1.0
    # Bonus for cells sitting on a map the *next* milestone needs.
    #
    # Depth heuristics assume the critical path only runs forwards. It does not: Oak's
    # Parcel must be carried from Viridian back to Pallet Town, and Viridian's north exit
    # stays blocked until it is. A forward-only bias sent 97% of restores to Viridian and
    # 0.5% to Oak's Lab, so the required action was effectively unreachable and the run
    # held the same milestone for 9.1M steps. Set above map_rank_weight * (rank spread)
    # so it can override the forward pull when the objective is behind the agent.
    target_bonus: float = 6.0
    # Fraction of restores drawn only from cells at the deepest milestone level.
    #
    # Milestone encodes irreversible world state, not just distance travelled. The gap
    # of one milestone between "carrying Oak's Parcel" and "delivered it" is the
    # difference between Viridian's north exit being shut and open, and the score
    # weights milestone at 1.0 against map_rank + target_bonus at 11 -- so 22 stale
    # pre-delivery Route 1 cells outvoted the 2 post-delivery ones and ~80% of restores
    # began in a world where the next milestone was unreachable. Measured: milestone 9
    # held for 2.0M steps.
    #
    # Not 1.0: the unrestricted remainder is how a frontier that turns out to be a dead
    # end gets abandoned.
    # Weight on the party health stored with a cell.
    #
    # A launch pad the agent cannot fight from is not a launch pad. Measured on the
    # milestone-11 frontier, the archived Viridian Forest cells sat at 10-40% party HP,
    # so every restore dropped the agent somewhere that fighting risked a wipe and
    # fleeing was the correct play. A scripted "mash A" policy won 60% of battles from
    # those same cells and levelled up; the learned policy declined, and was right to.
    # `level_sum` never exceeded 6.67 in 22M steps.
    #
    # 2.0 puts a full-health cell two milestone-levels ahead of a dead one, so health
    # outranks small map-rank differences without ever overriding the frontier
    # restriction or the target bonus.
    hp_weight: float = 2.0
    # Flat penalty for a cell the agent cannot act from. Health as a *gate*, not a nudge.
    #
    # `hp_weight` alone was measured to be far too weak to matter: it contributes at most
    # 2.0 against a milestone-plus-map_rank spread of ~25, so a dying frontier cell
    # always outranked a full-health one a map back. The consequence, sampled from the
    # live archive at 79.8M env steps: **75% of episodes started below 0.3 HP and 55%
    # started essentially dead**, with 47% of all restores landing in Pewter City where
    # 0 of 24 cells were above 0.33 HP. An episode that begins one hit from a blackout
    # cannot fight, cannot level and cannot heal, which is why `level_sum` sat at 8 and
    # `heal_visit`, `ball` and `party_member` had still never fired 20.8M steps after
    # being made reachable.
    #
    # Implemented as a *filter* on the candidate set rather than a score penalty. A
    # penalty has to be tuned against the depth terms and cannot win: measured on the
    # milestone-13 frontier, every deep cell was dying (Pewter, 24 cells, all under
    # 0.33 HP) and every healthy one was a post-blackout Route 1 cell seven map-ranks
    # behind (20 cells). At a penalty of 4.0 the Route 1 group simply outweighed the
    # 2 healthy North Gate cells by count and took 66% of restores; at 2.0 the dying
    # Pewter cells still won. Removing non-viable cells from the candidate set instead
    # leaves the depth ordering exactly as it was and lets the deepest *survivable*
    # cell win on its own merits.
    require_viable: bool = True
    # Cells per Pokemon Center / Poke Mart shielded from the per-level eviction cap.
    # See `FrontierArchive._enforce_level_caps_locked` for why they need shielding at all.
    utility_cells_per_map: int = 4
    # Strongest cells (by party experience) shielded from the per-level eviction cap.
    #
    # Eviction ranks victims by `map_rank` and had no strength term, so a level-10 cell
    # on Route 1 died before a level-6 cell on Route 2. That silently undid the whole
    # point of ratcheting the archive on experience: measured an hour after the ratchet
    # landed, the trim triggered by reaching milestone 14 took the archive's best party
    # from 722 XP (level 10) back down to 327 (level 8) -- every stronger state gone.
    # Strength is the scarcest thing this run accumulates and it is not recoverable by
    # revisiting a map, unlike coverage.
    strongest_cells_kept: int = 8
    # Fraction of restores reserved for the strongest cells in the archive, regardless
    # of depth.
    #
    # Strength accumulates *behind* the frontier -- the agent gets strong by grinding
    # somewhere safe, and the states that carry the experience are typically
    # post-blackout Pallet Town or Route 1 cells at map_rank 1-5. Selection is
    # depth-first, so they are never used: sampling the live archive at 87M env steps,
    # **99.5% of restores drew a 327 XP cell and 0.5% drew anything stronger**, with 89%
    # landing in Pewter City. The archive banked experience into cells it then refused
    # to launch from, so strength could never compound and the party stayed at level 8
    # against a level-14 Onix.
    #
    # This is a sampling reservation rather than another score term because a score term
    # provably cannot win here: closing the measured 11-point map_rank gap would need a
    # weight around 150, which would make experience dominate depth entirely. The same
    # lesson came out of the viability work -- selection weight depends on cell *count*
    # as much as on score, so gates and reservations behave predictably where weights do
    # not.
    #
    # Raised 0.15 -> 0.35 on measurement. At 0.15 the reservation demonstrably worked --
    # the top-experience cells went from `chosen = 0` to 2-4, and Pallet Town took 12% of
    # restores -- but it was not enough to give the agent anywhere to train. Sampling the
    # live archive at 88.3M env steps: Pewter City 60%, Pewter Gym 20%, and **only 7.4%
    # of restores landed on a map with wild encounters at all**. Pewter is a town, so it
    # has no grass, and the single battle available there is a level-14 Onix the party
    # cannot beat at level 8. Four rollouts from live restores spent 99% of 4800 steps in
    # Pewter City and entered zero battles.
    #
    # The frontier being a place where nothing can happen is exactly when depth-first
    # selection needs overriding. One restore in three now goes to the strongest states
    # instead, which sit in Pallet Town and on Routes 1 and 2 -- next to the grass the
    # run needs in order to make Brock winnable.
    strength_prob: float = 0.35
    # How many of the top cells by experience the reservation draws from.
    strength_pool: int = 8
    # Weight on -log1p(visits): prefers cells the agent rarely manages to reach.
    #
    # The only term that distinguishes cells *within* one map. Everything else --
    # milestone, map_rank, target bonus -- is constant across a map, so in a maze the
    # size of Viridian Forest nothing pulled restores toward the far exit, and the run
    # held milestone 11 for 3.9M steps while covering only the southern half. Measured
    # there: 85.9 mean visits near the entrance against 25.5 deep inside,
    # corr(y, visits) = +0.73.
    #
    # Raised 0.5 -> 1.0 on measurement: at 0.5 the deepest quartile of Viridian Forest
    # cells took only 34% of draws against 26% of the population -- 1.33x enrichment,
    # too weak to thread a maze while milestone 11 sat unmoved for 5.9M steps.
    #
    # The ceiling is not what I first assumed. Even at 0.5 the score spread across the
    # frontier is 2.16, already past the 1.0-per-level milestone gap, so a rarely visited
    # shallow cell could outscore a frontier one either way. What actually protects the
    # frontier is structural -- `frontier_prob` restricts 80% of draws to the deepest
    # level before any score is computed -- so the weight trades against nothing.
    #
    # At 1.0 a cell reached 8 times outranks one reached 149 times by e^2.8 ~ 16x.
    visit_weight: float = 1.0
    frontier_prob: float = 0.8
    # Widen the frontier set to shallower levels until it holds at least this many
    # cells, so a freshly opened milestone level does not funnel 80% of episodes
    # through its single first save state.
    frontier_min_cells: int = 6
    # Widen past the deepest level until at least one cell has this much party HP.
    #
    # `hp_weight` ranks cells *within* the frontier set, but `frontier_prob` chooses that
    # set first -- so when the whole deepest level is nearly dead, the preference has
    # nothing to work with. Measured in Viridian Forest: all 82 cells held one level-6
    # Pokemon at 10-40% HP against trainers fielding level 6-9 teams, so every restore
    # was a loss and the party could never level out of it.
    #
    # 0.6 rather than 1.0: demanding full health would reject a frontier that is merely
    # scratched, and the shallower level it falls back to costs real progress.
    frontier_min_hp: float = 0.6
    directory: str = str(REPO_ROOT / "runs" / "archive")


@dataclass
class LLMConfig:
    """Ollama-backed subgoal proposer. Always asynchronous: never blocks an env step."""

    enabled: bool = True
    host: str = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

    # Measured on this machine (see docs/ARCHITECTURE.md "Choosing the local model"):
    # qwen3:8b honours Ollama's `think: false` and returns clean JSON in ~20 tokens.
    # qwen3-vl:8b ignores it, spends its entire budget in the `thinking` field, and at
    # 6.1 GB cannot co-reside with the 3.7 GB training process on a 6 GB card. Vision
    # buys little here regardless: the symbolic state summary is strictly more precise
    # than a 160x144 screenshot for choosing a subgoal.
    model: str = "qwen3:8b"
    vision_model: str = "qwen3-vl:8b"
    use_vision: bool = False  # opt-in; forces the proposer onto CPU

    # The proposer runs on CPU by default so it never competes with training for VRAM.
    # num_gpu=0 measured 10.3 tok/s vs 15.4 partially offloaded -- a small loss, and it
    # removes the risk of an OOM mid-run.
    num_gpu: int = 0
    # Cap on llama.cpp worker threads.
    #
    # Left unset, Ollama sizes its pool to the machine and measured 518% CPU -- five of
    # sixteen cores, held continuously (see `min_interval_s`). The eight emulator workers
    # are pure CPU and were starved by it: collection ran at 300 env steps/s against the
    # ~760 the replay ratio permitted, and against the ~1200 recorded in docs/PROOF.md.
    # The proposer is off the critical path, so it is the right thing to bound.
    num_thread: int = 4
    keep_alive: str = "30m"

    # Minimum wall-clock seconds between requests.
    #
    # The proposer is single-flight but had no cooldown, so `_loop` reissued the moment a
    # worker became eligible -- at 8 workers and ~2k steps of eligibility that is always
    # true, and inference ran back-to-back at 100% duty cycle forever. A subgoal is
    # coarse guidance refreshed over minutes; nothing about it needs a continuously hot
    # 8B model. At 30 s this drops the duty cycle from ~100% to ~25%.
    min_interval_s: float = 30.0

    # Minimum env steps a worker must accumulate before it is eligible for a refresh.
    # The proposer is single-flight and round-robins, so the real rate is set by model
    # latency, not by this number.
    refresh_steps: int = 2_048
    # 120 s was absurd for a ~20-token reply and turned every shutdown into a stall:
    # `proposer.stop()` runs in the trainer's `finally`, so a hung daemon left the
    # process alive for minutes after "finished" -- with `scripts/train.sh stop`
    # budgeting only 180 s before SIGKILL.
    #
    # Not lower than this: the *first* request after the model is evicted pays a cold
    # start, loading ~5 GB of qwen3:8b from disk, measured at 47.9 s, and a long
    # generation runs to ~48 s more. Steady state is ~8 s once keep_alive holds the model
    # resident, so this budget is only ever spent on the cold path.
    #
    # The shutdown stall it was blamed for is fixed properly in `SubgoalProposer.stop`,
    # which closes the HTTP session so an in-flight read fails immediately rather than
    # being waited out. This value only has to stay comfortably inside the 180 s that
    # `scripts/train.sh stop` allows before SIGKILL.
    request_timeout: float = 90.0
    max_tokens: int = 192
    num_ctx: int = 4096
    temperature: float = 0.6
    # If the daemon is unreachable, fall back to the scripted milestone curriculum
    # rather than crashing a 24 h run.
    hard_fail: bool = False
    cache_path: str = str(REPO_ROOT / "runs" / "llm_cache.jsonl")


@dataclass
class TrainConfig:
    num_envs: int = 8
    # Step budget for the whole run.
    #
    # Raised 20M -> 200M because 20M was silently the binding constraint on progress: the
    # run reached 20,000,056 steps at milestone 11/46, exited cleanly, and the supervisor
    # correctly read rc=0 as "done" rather than as a crash to restart. Nothing was wrong;
    # the budget simply ran out, and from the outside that is indistinguishable from a
    # plateau.
    #
    # Sizing: milestones have been costing ~2.9M steps each, so the remaining 35 need
    # ~100M. 200M leaves headroom for the gyms and dungeons, which are harder than
    # anything reached so far. At the measured 470 steps/s this is ~118 hours -- the
    # budget is deliberately not the thing that ends the run.
    total_steps: int = 200_000_000
    # Replayed steps per collected step. The collector throttles itself to hold this
    # ratio, so the learner is never starved and the buffer never becomes stale.
    # 2.0 keeps 8 emulators at ~60% duty while the GPU stays saturated.
    replay_ratio: float = 2.0
    prefill: int = 5_000
    eval_every: int = 50_000
    # Checkpoint on *either* trigger. The step trigger alone is a trap: at CPU speed
    # 20k env steps is ~20 minutes, so an interrupted run could lose everything.
    checkpoint_every: int = 20_000  # env steps
    checkpoint_every_s: float = 300.0  # wall-clock seconds
    log_every: int = 1_000
    seed: int = 0
    # Stall detection. The run watches its own progress signals and, when nothing that
    # counts as progress has moved for `stall_window` env steps, actively probes whether
    # the next milestone is reachable at all (see pokewm/agent/stall.py). Costs one
    # short-lived emulator and a few seconds, and is the difference between noticing a
    # stall in minutes rather than after 9M steps.
    stall_window: int = 750_000
    # Longer window for *game* progress alone (milestone / story flags). Secondary
    # counters like unique_coords never stop climbing, and under a single window they
    # masked a 2.0M-step milestone plateau as "healthy".
    stall_hard_window: int = 3_000_000
    stall_check_every: int = 100_000
    # How often to recompute which maps the archive should aim restores at.
    #
    # Not merely a refresh cadence. When the next milestone names an unreached map the
    # archive falls back to the deepest map it currently holds, and that depends on the
    # archive rather than on the milestone -- so computing it only on milestone
    # transitions samples it at precisely the wrong instant, the step the agent first
    # enters the new map and before any cell there exists.
    target_refresh_every: int = 50_000
    stall_probe: bool = True
    stall_probe_cells: int = 6
    stall_probe_steps: int = 4_000
    device: str = "cuda"
    amp: bool = True
    # Torch intra-op threads. On CPU, letting torch grab every core starves the emulator
    # subprocesses and collection collapses (measured: 24 env-steps/s unthrottled).
    # 0 leaves torch's default alone, which is right when the learner is on the GPU.
    torch_threads: int = 0
    compile: bool = False
    logdir: str = str(REPO_ROOT / "runs" / "default")


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    wm: WorldModelConfig = field(default_factory=WorldModelConfig)
    ac: ActorCriticConfig = field(default_factory=ActorCriticConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    archive: ArchiveConfig = field(default_factory=ArchiveConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def preset(name: str) -> "Config":
        cfg = Config()
        if name == "laptop":
            return cfg
        if name == "cpu":
            # No-GPU fallback. The world model is cut to ~4M parameters so a CPU
            # gradient step stays under a second, leaving enough cores for the
            # emulator farm. Learns navigation and the early milestone chain; too
            # small for the full game.
            cfg.wm = replace(
                cfg.wm,
                deter=256,
                stoch=16,
                classes=16,
                hidden=256,
                cnn_depth=16,
                layers=2,
                ensemble_size=3,
                horizon=10,
                batch_size=16,
                batch_length=16,
                imag_batch=64,
                bins=127,
            )
            cfg.ac = replace(cfg.ac, hidden=256, layers=2)
            cfg.replay = replace(cfg.replay, capacity=120_000, min_size=2_000)
            cfg.train = replace(
                cfg.train,
                device="cpu",
                amp=False,
                num_envs=6,
                prefill=2_000,
                replay_ratio=1.0,
                torch_threads=6,  # leaves 10 cores for the 6 emulator subprocesses
            )
            return cfg
        if name == "smoke":
            cfg.env = replace(
                cfg.env, max_episode_steps=64, frame_stack=2, action_frames=8
            )
            cfg.wm = replace(
                cfg.wm,
                deter=64,
                stoch=8,
                classes=8,
                hidden=64,
                cnn_depth=8,
                layers=1,
                ensemble_size=2,
                horizon=4,
                batch_size=2,
                batch_length=8,
                imag_batch=8,
                subgoal_dim=8,
                bins=51,
            )
            cfg.ac = replace(cfg.ac, hidden=64, layers=1)
            cfg.replay = replace(cfg.replay, capacity=2_000, min_size=16)
            cfg.train = replace(
                cfg.train,
                num_envs=1,
                total_steps=256,
                prefill=32,
                replay_ratio=2.0,
                device="cpu",
                amp=False,
                log_every=16,
                checkpoint_every=10_000_000,
                eval_every=10_000_000,
            )
            cfg.llm = replace(cfg.llm, enabled=False)
            cfg.archive = replace(cfg.archive, max_cells=32)
            return cfg
        raise ValueError(f"unknown preset {name!r}; expected 'laptop', 'cpu' or 'smoke'")
