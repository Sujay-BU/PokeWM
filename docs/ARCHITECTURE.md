# Architecture and design rationale

Every non-obvious choice here was made against one of three constraints: a **6 GB laptop
GPU**, a **~10⁵-step horizon**, and the requirement that a **world model be the core** of
the agent rather than an accessory. Where a decision was settled by measurement, the
measurement is given.

---

## 1. The shape of the problem

Pokémon Red is not hard because its dynamics are hard. It is hard because of the
horizon-to-reward ratio:

| | value |
|---|---|
| Agent steps to Hall of Fame (expert) | ~1.2×10⁵ |
| Distinct irreversible progress events | ~320 story flags, 8 badges |
| Longest gap between forced progress events | thousands of steps |
| Critic effective horizon at γ=0.997 | ~333 steps |

The gap between the last two rows is the entire problem. Three families of solution exist,
and the design uses all three at different levels:

1. **Shorten the effective horizon** — dense exploration shaping (§5), so the credit-
   assignment chain the critic must span is O(10) steps, not O(10³).
2. **Shorten the actual horizon** — the frontier archive (§4), which lets an episode start
   at the frontier so it only has to solve *one* milestone.
3. **Amplify the data** — a world model (§3), so policy learning is bounded by model
   accuracy rather than by emulator throughput.

---

## 2. Observation design

```
frame     (5, 72, 80) uint8    4 stacked luma frames + 1 visited-tile overlay
symbolic  (22,)       float32  decoded RAM
subgoal   (24,)       float32  one-hot, from the LLM proposer
```

**Why multi-modal and not pixels-only.** A pixel-only observation forces the world model
to spend capacity re-deriving facts that are exactly representable in a handful of numbers
— badge count, party HP, whether a text box is open. Simulus (2025) makes the general
argument for mixed tokenisation; here it is unusually clear-cut because the Game Boy's RAM
*is* the game state and reading it is free. The 22 symbolic features are listed in
`ram_map.SYMBOLIC_FEATURES` and every one is scaled into roughly [0,1] so no
normalisation statistics have to be tracked as the agent reaches new parts of the game.

**Why the visited-tile overlay.** Game Boy screens are locally ambiguous: one patch of
grass looks like every other. Without a spatial memory the POMDP requires the recurrent
state to carry a full map, which is a large ask of a 1024-unit GRU. The overlay renders a
36×40-tile crop of "where have I already stood on this map", centred on the player. PokeRL
(2026) measured **+40.6% unique tiles visited** from exactly this channel, and it costs one
uint8 plane.

**Why 72×80.** The screen is 144×160; halving keeps the aspect ratio and every tile
distinguishable. 72 is divisible by 8 but not 16, which is why the conv stack uses three
stride-2 stages rather than four — four would not invert cleanly in the decoder. That
constraint is enforced by `nets.check_frame_shape` and tested.

**Why START is in the action space.** PokeRL removes it to stop menu spam. We keep it: the
late game is unreachable without it — HM field moves, healing items and the bicycle all
live behind the START menu. Menu spam is handled by the step cost and the epistemic bonus
instead of by amputating the action space, because an action space that cannot finish the
game makes the proof in `PROOF.md` vacuous.

---

## 3. The world model

A **discrete-latent RSSM** in the DreamerV3 lineage, with the post-2023 additions that
replicated.

```
h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})          deterministic path, 1024 units
z_t ~ q(z_t | h_t, x_t)                        posterior, 32 categoricals x 32 classes
ẑ_t ~ p(z_t | h_t)                             prior -- this is what imagination uses
```

**Why discrete latents.** The observable state genuinely is discrete — tile grids, menu
indices, integer HP. Categorical posteriors do not suffer the collapse that Gaussian
latents show on sharply multi-modal transitions such as a screen fade or a battle intro,
which this game is full of. This is Hafner et al.'s argument for Atari and it applies more
strongly here.

**Loss.**

```
L = L_frame + L_symbolic + L_reward + L_cont + β·L_KL + L_ensemble
```

| component | choice | why |
|---|---|---|
| reward head | two-hot classification over 255 symlog bins | Rewards span 0.002 (step cost) to 64 (badge). MSE regression lets the badge gradient swamp everything for thousands of updates. Classification is stable under heavy-tailed targets (Imani & White 2018; Farebrother et al. 2024). |
| KL | free bits 1.0, dyn 0.5 / rep 0.1 | Weighting the dynamics term 5× the representation term is what keeps imagination rollouts on-manifold. Note free bits *block the prior's gradient entirely* while KL sits at the floor — verified explicitly in `test_wm.py`. |
| latents | 1% unimix | Keeps the KL finite and gives the exploration floor that `PROOF.md` Lemma 1 depends on. |
| ensemble | 4 heads, JSD disagreement | §6. |

**Sizing (measured, RTX 4050 6 GB):** 59.4 M parameters, 3.7 GiB peak VRAM, feature
dimension 2048.

---

## 4. The frontier archive

The single most important component, and the subject of `PROOF.md` Lemma 2.

A cell is keyed by three things together:

```
progress_key            hash of badges, Pokedex count, party size, 320-byte flag block
milestone index         which link of the critical path has been reached
(map, x//8, y//8)       coarse position bucket
```

Each component was added because the archive stopped working without it, and both
additions were found by watching live runs rather than by testing:

* **milestone** — everything between leaving the bedroom and receiving a starter sets no
  story flag, so a progress-only key pinned the archive to a single bedroom cell.
* **coarse position** — everything between "on Route 1" and "reached Viridian City" sets
  no story flag *and* crosses no milestone. Without position the archive froze at **28
  cells for 2.7 million steps**, and with it frozen, Go-Explore is simply switched off:
  the frontier cannot advance inside any long phase. Cells resumed growing (28 → 44
  within three minutes) the moment position entered the key.

The 8-tile bucket is the compromise. Keying on raw (x, y) really would produce tens of
thousands of near-identical cells per town; bucketing gives roughly 25 cells for a large
map, which the `max_cells` cap and the novelty-weighted softmax handle comfortably. This
is also what Go-Explore actually does — its cells are a downsampled state, not a bare
progress flag.

Selection is softmax over

```
score(c) = milestone_index(c) + w / sqrt(1 + times_chosen(c))
```

The first term drives the frontier forward; the second is the standard count-based bonus
(Strehl & Littman 2008) and stops the archive collapsing onto one deep dead end. Eviction
never removes the unique deepest cell, which is what makes the frontier monotone and hence
makes assumption (A2) of the proof a theorem rather than a hope.

**Why this works here specifically.** The emulator is deterministic, so "return to a
previously reached state" is a save-state load — *exact*, not an approximate return policy.
Go-Explore's hard case (stochastic return) does not arise.

**The cell key includes the milestone index**, not just the progress hash. This was found
by watching a live run: the archive sat at a single cell for the whole opening, because
everything between leaving the bedroom and receiving a starter sets *no story flag*, so
the progress hash never changed and the one archived cell stayed pinned to the bedroom.
Keying on `(progress_key, milestone)` lets the frontier advance within a flag-free phase,
which is exactly what Lemma 2 of `PROOF.md` needs ("a cell at each level $i$"). Guarded by
`tests/test_env.py::TestVecEnv::test_cell_key_advances_with_the_milestone`.

---

## 4b. Reading RAM that is mid-write

The emulator is sampled between frames, so multi-byte structures can be observed
part-written. This produced the single most damaging bug in the project's history and is
worth stating as a general rule: **a zeroed field means "not yet written", not "zero".**

`wPartyCount` is incremented *before* the Pokemon's stats are copied into the party
struct. For a few frames after receiving the starter, slot 0 reads back as
`species=0, level=0, hp=0, max_hp=0`. The original `party_wiped` check —
`all(not m.alive for m in party)` — read that as a total blackout and terminated the
episode at the exact instant the agent received its first Pokemon. The agent then
restored from the archive to a point before the starter, walked back to Oak, received it
again, and terminated again: an endless loop.

The blast radius was the whole project. Every milestone past `got_starter` was
unreachable, `party_level_sum` sat at exactly 0 for 4.6 million steps, and the world
model's continue-head learned the falsehood "a party appearing means the episode ends".
The stall it caused was misread twice as an exploration problem before the actual cause
was found.

The fix is `GameState.live_party`, which filters to slots with `max_hp > 0`; every
party-derived property is computed from it. The same hazard applies anywhere a count and
its payload are written separately, which is why `party_hp_frac` and `party_level_sum`
use it too. Guarded by `test_ram_map.py::TestGameState` (unit) and
`test_env.py::TestRewards::test_acquiring_a_pokemon_does_not_terminate_the_episode`
(end-to-end).

## 5. Reward design

First-visit credit on monotone statistics, so that no term can be farmed:

| term | weight | class |
|---|---|---|
| story flag | +4.0 each | potential-based (§6.1 of PROOF.md) |
| badge | +64.0 each | potential-based |
| party level | +0.6 each | potential-based |
| Pokédex owned / seen | +2.0 / +0.2 | potential-based |
| new map / new tile | +3.0 / +0.02 | summable novelty |
| epistemic (JSD) | ×0.10 | bounded by 0.1·ln4 |
| LLM subgoal | +3.0, once per assignment | verification-gated |
| step cost | −0.002 | constant |

The dense `new_tile` term is doing quiet but essential work: it is what makes the *reward*
horizon O(10) steps even though the *milestone* horizon is O(10³).

---

## 6. Intrinsic motivation: disagreement, not error

The exploration bonus is the **Jensen–Shannon divergence across a 4-member ensemble** of
one-step latent predictors:

```
JSD = H(mean_i p_i) − mean_i H(p_i) ∈ [0, ln 4]
```

Zero iff the members agree, bounded by ln N so it cannot explode.

The alternative — prediction error (ICM-style) — fails on this game specifically. Pokémon
Red is full of transitions that are stochastic but perfectly well understood after a
handful of samples: wild encounter rolls, damage variance, critical hits. A prediction-error
bonus chases those forever (the noisy-TV problem, Pathak et al. 2017). Disagreement
distinguishes *epistemic* uncertainty from *aleatoric*: once the ensemble has seen enough
encounter rolls its members agree on the distribution, and the bonus decays. Plan2Explore
(Sekar et al. 2020) established this for latent world models; Simulus (2025) is the
version adopted here, including the prioritised world-model replay that shares its
motivation.

---

## 7. Choosing the local model

The task asked for the best local Ollama model. Both candidates were measured on this
machine:

| model | size | `think:false` honoured | warm latency | verdict |
|---|---|---|---|---|
| `qwen3:8b` | 5.2 GB | **yes** — clean JSON in ~20 tokens | **~7 s** | **selected** |
| `qwen3-vl:8b` | 6.1 GB | **no** — spends the whole budget in the `thinking` field, returns empty `content` | >60 s, often times out | rejected as default |

`qwen3-vl:8b` also cannot co-reside with the 3.7 GiB training process on a 6 GB card, and
swapping between the two models costs ~30 s each way. Vision buys little here regardless:
the symbolic state summary (exact coordinates, badges, party HP, story-flag count) is
strictly more precise for choosing a subgoal than a 160×144 screenshot would be. The
vision path remains available behind `--llm-vision` and forces CPU execution.

The proposer runs with `num_gpu: 0` by default — 10.3 tok/s on CPU versus 15.4 partially
offloaded, a small loss that removes any chance of an OOM taking down a 24-hour run.

### How the LLM is wired in

**The RL loop never waits for the LLM.** One background thread, one in-flight request,
round-robin over the workers that have gone longest without a refresh. Whatever throughput
the local model achieves is the refresh rate. If it achieves zero — daemon down, model
evicted, machine busy — every worker keeps its last subgoal and training is unaffected.
`build_proposer` degrades to a `NullProposer` if Ollama is unreachable at startup.

Concurrency was rejected deliberately: parallel requests to one Ollama daemon serialise
anyway, and queueing them only builds latency between the state a subgoal was chosen for
and the state it is applied to.

**Measured behaviour** on four hand-built scenarios after prompt tuning:

| scenario | proposed | latency |
|---|---|---|
| wild battle in progress | `WIN_BATTLE` | 7.0 s |
| healthy party, Pewter, no badge | `CHALLENGE_GYM` | 7.9 s |
| opening state, Pallet Town | `MAIN_QUEST` | 6.3 s |
| hurt party (one fainted), Cerulean | `CHALLENGE_GYM` *(HEAL would be better)* | 27.7 s |

Three of four are sensible; the fourth is wrong. That is the realistic ceiling of an 8B
model here, and it is exactly why the subgoal bonus is **verification-gated**: a wrong
suggestion simply never pays out. See `PROOF.md` §6.4.

---

## 8. Throughput

The learner is GPU-bound and the emulator farm is CPU-bound, so they run concurrently: a
collector thread drives 8 headless PyBoy subprocesses while the main thread takes gradient
steps. The collector self-paces against a target replay ratio (replayed steps per collected
step, default 2.0) so neither side starves the other.

Measured on the RTX 4050 / 16-core host:

| configuration | updates/s | replay steps/s | peak VRAM |
|---|---|---|---|
| B=16, T=64, fp32, imagine 1024 | 1.22 | 1 250 | 4.41 GiB |
| B=16, T=64, bf16, imagine 256 | 1.85 | 1 893 | 3.70 GiB |
| **B=32, T=32, bf16, imagine 256** | **2.30** | **2 360** | **3.70 GiB** |

The 32×32 geometry replays the same 1024 transitions per update as DreamerV3's 16×64 but
halves the sequential depth of the RSSM filtering loop, which on a laptop GPU is
kernel-launch bound. That single change is worth 24%.

Emulator throughput *in isolation*: **657 steps/s** for one headless worker, **1874
steps/s** aggregate across 8. Serialising collection and training instead of overlapping
them measured 1.43 updates/s versus 2.30 — hence the threaded collector.

**End-to-end is much lower than either number in isolation, and that is the honest figure
to plan with.** With the learner running, the collector's policy forward has to queue
behind the learner's kernels on a single small GPU:

| parallel envs | env-steps/s | updates/s | worker PSS |
|---|---|---|---|
| 8 | 213 | 1.60 | ~2.3 GB |
| 12 | ~330 | ~1.45 | ~3.5 GB |
| 16 | 438 | 1.34 | ~4.6 GB |

Scaling is nearly linear in worker count because the per-iteration cost is dominated by
the collector's single batched GPU forward, not by emulation — so more emulators amortise
the same forward. The run ships at **12**, which is the point where memory still leaves
headroom on a 16 GB machine.

Note the implied replay ratio is ~5-8x, not the configured 2.0: `replay_ratio` is an upper
bound on collection, and on this hardware the system is *collection-bound*, so the throttle
never engages. On a machine with a bigger GPU the throttle would start to matter.

**The epistemic bonus rides along with the action** in a single IPC exchange rather than
taking its own synchronous round trip. At roughly 1 ms of emulation per step, a second
round trip across 8 pipes cost more than the emulation it was annotating. Measured 1749
steps/s aggregate *while a training job was competing for cores*.

**The model lock guards writes only.** The collector reads network parameters to choose
actions; the learner writes them. The obvious implementation — hold a lock across the
whole `train_step` — is catastrophic: it pins the lock for ~430 ms of every ~435 ms cycle,
so the collector runs only in the gaps and collection collapses from ~1200 to **36
env-steps/s**. This was invisible in every unit test and only showed up as a suspicious
`sps` column in a live GPU run.

The fix follows from asking what actually races. Forward and backward only *read* weights
(backward writes `.grad`, which the collector never touches); the sole writers are
`optimizer.step()` and the slow-critic EMA. Guarding just those takes the hold time to a
few milliseconds. `test_trainer.py::TestTrainerRun::test_model_lock_is_not_held_across_the_whole_update`
instruments the lock and asserts it is held for under 50% of an update.

### Worker start method

Workers use **`forkserver`**, not `spawn`. With `spawn`, every worker re-imports
`__main__` — which for a training run is `pokewm.train`, and therefore pulls in torch and
the CUDA runtime. Measured at **527 MB RSS per worker**: 16 emulators cost 8.4 GB on a
16 GB machine, most of the way to an OOM kill. `forkserver` starts one lean server
(measured 35 MB) that imports only `pokewm.emulator.vec_env` and forks workers from it.

Plain `fork` would share more still, but forking a process that has already initialised a
CUDA context is unsafe; forkserver avoids that because the server is a separate process
that never touches CUDA.

### Checkpointing and shutdown

Checkpoints fire on **either** an env-step count or a wall-clock interval (default 300 s).
The step trigger alone is a trap: at CPU speed 20 000 env steps is ~20 minutes, and an
interrupted run could lose everything. Discovered the hard way on a live run that had to be
SIGKILLed with no checkpoint on disk.

Shutdown installs explicit SIGINT **and SIGTERM** handlers that set a flag the training
loop polls, rather than relying on `KeyboardInterrupt`. SIGTERM — which is what process
managers actually send — never raises it, and the main thread spends most of its time
inside long C-level torch calls where even SIGINT is delayed. With the handler, a clean
stop-with-checkpoint takes ~6 s. Covered by
`test_trainer.py::TestTrainerRun::test_shutdown_flag_stops_the_loop_and_checkpoints`.

`scripts/train.sh supervise` wraps this in a restart loop for multi-day runs: if the
trainer dies for a reason that is not a bug in the agent — an OOM kill, a GPU driver
fault, a transient CUDA error — it relaunches from the last checkpoint, losing at most one
checkpoint interval. A deliberate `stop` removes a flag file so the supervisor does not
fight it. (The first version of this was silently useless: `set -e` aborted the supervisor
the instant `wait` reported the trainer's non-zero exit, i.e. in precisely the case it
existed to handle. Verified by killing a live trainer with SIGKILL and confirming
relaunch.)

### The `cpu` preset

A ~7 M-parameter fallback for machines with no usable GPU: `deter=256`, `stoch=16×16`,
batch 16×16, imagination batch 64, 6 workers. It also caps `torch.set_num_threads(6)` —
letting CPU torch grab all 16 cores starves the emulator subprocesses and collection
collapses to 24 env-steps/s. It learns the early milestone chain but is too small for the
full game.

---

## 9. The intro is scripted, and why that is legitimate

Everything before the player gains control — logo, title menu, two name-entry screens,
~90 dialogue boxes — is a fixed, information-free prefix. The name-entry keyboard alone is
a ~200-step combinatorial detour that would dominate early training and teaches nothing.
Every published Pokémon Red agent (PokemonRedExperiments, pokegym, PokeRL) starts from a
canned post-intro state.

This repository does **not** ship one. `pokewm/emulator/bootstrap.py` derives it from the
ROM on first run using only button presses and RAM reads: it mashes A, detects the
4-entry naming menus (the only 4-entry menus reachable during the intro) and picks the
preset names, then confirms control by moving the sprite and checking the coordinate
actually changed. Takes 0.48 s, is fully reproducible and auditable, and the learned
policy's episode begins exactly where control begins.

---

## 10. File map

```
pokewm/
  config.py               all tunables; one serialisable object per run
  emulator/
    ram_map.py            WRAM decoding -> GameState -> 22-D symbolic vector
    maps.py               the 248-entry map id table from pret/pokered
    bootstrap.py          scripted traversal of the title/naming screens
    env.py                Gymnasium env: observation, reward, save states
    vec_env.py            8 subprocess workers
    archive.py            Go-Explore frontier archive
  wm/
    nets.py               symlog, two-hot, KL, lambda-returns, conv stacks
    rssm.py               recurrent state-space model
    world_model.py        encoder + RSSM + heads + epistemic ensemble
    actor_critic.py       imagination actor-critic
    replay.py             prioritised sequence replay
  llm/
    subgoals.py           24-subgoal vocabulary + verification predicates
    ollama_client.py      thin Ollama chat wrapper
    proposer.py           async single-flight proposer thread
  agent/
    milestones.py         the 46-milestone critical path
    trainer.py            concurrent collector + learner, checkpointing
  train.py                training entry point
  play.py                 live GUI viewer
```
