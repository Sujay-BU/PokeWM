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
symbolic  (32,)       float32  decoded RAM
subgoal   (24,)       float32  one-hot, from the LLM proposer
```

**Why multi-modal and not pixels-only.** A pixel-only observation forces the world model
to spend capacity re-deriving facts that are exactly representable in a handful of numbers
— badge count, party HP, whether a text box is open. Simulus (2025) makes the general
argument for mixed tokenisation; here it is unusually clear-cut because the Game Boy's RAM
*is* the game state and reading it is free. The 32 symbolic features are listed in
`ram_map.SYMBOLIC_FEATURES` and every one is scaled into roughly [0,1] so no
normalisation statistics have to be tracked as the agent reaches new parts of the game.

### 2b. Menus are state, and encoding them badly is worse than not encoding them

Seven of the 32 features exist only to make menus legible, and they replaced a feature
that was actively lying. This is worth its own subsection because the failure mode
generalises: **Gen 1 menu RAM is stale-by-default**, and a byte that is never cleared
reads as a signal while being a constant.

* **`menu_active` was a constant.** It was defined as `max_menu_item > 0`, but
  `wMaxMenuItem` is never cleared when a menu closes — measured nonzero in **1119 of 1119**
  archived overworld cells with no menu open and no script running. The feature was
  removed, not fixed.
* **The battle menu is a 2×2 grid, not a list.** `wMaxMenuItem = 1` carries only the row;
  the column lives in `wTopMenuItemX` alone. The old encoding was `menu_item /
  max_menu_item`, so **FIGHT and ITEM were both 0.0** and PKMN and RUN were both 1.0 — the
  agent could not distinguish attacking from opening the bag. `menu_row_origin` and
  `menu_col_origin` (`wTopMenuItemY` / `wTopMenuItemX`) carry it, and they also separate
  the battle menu (y=14) from the move list (y=12) from a text box (y=2), which matters
  because nested menus shift `wMaxMenuItem` 7 → 1 → 3.
* **Shop lists scroll.** `wCurrentMenuItem` indexes a *row on screen*, not a list entry —
  it saturates at the last visible row while `wListScrollOffset` pages beneath it.
  Measured: three different Poké Mart items all read `item = 2`, differing only in scroll.
  `list_index = menu_item + list_scroll` is the absolute entry.
* **The quantity selector moves no menu byte.** `wItemQuantity` steps 1, 2, 3… while every
  other menu byte holds still; buying one Poké Ball and buying five were identical
  observations.

One ambiguity is left deliberately: an item list with the cursor on entry 0 and a
just-opened quantity selector at qty = 1 are *the same RAM state*. No feature derived from
these bytes can separate them — the frame differs (the selector draws a box), so it is the
pixel path's job.

**Verification recipe, reusable:** drive every menu in the game, collect the distinct
menu-RAM tuples, and assert that distinct tuples map to distinct encoded vectors. 22 of 22
with zero collisions as of this change. Before trusting any Gen 1 menu byte as a live
signal, check it against overworld states with no menu open.

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

The 4-tile bucket is the compromise. Keying on raw (x, y) really would produce tens of
thousands of near-identical cells per town; bucketing gives a few dozen cells for a large
map, which the `max_cells` cap and the novelty-weighted softmax handle comfortably. This
is also what Go-Explore actually does — its cells are a downsampled state, not a bare
progress flag.

Selection is softmax over

```
score(c) = milestone(c)
         + map_rank_weight · max(map_rank(c), 0)      # 1.0
         + target_bonus   if c is on a target map     # 6.0
         + hp_weight · hp_frac(c)                     # 2.0
         − visit_weight · log1p(visits(c))            # 1.0
         + novelty_weight / sqrt(1 + chosen(c))       # 1.0
```

Every term after the first was added because the archive demonstrably stopped working
without it, and each is documented with its measurement in `Cell.score`'s docstring:

* **`map_rank`** — `milestone` is a monotone *history* counter, so a state that reached
  Route 1 and then wandered home still scores milestone 6, identical to a state actually
  standing on Route 1. Measured: of 62 cells at the frontier level, only 12 were on
  Route 1 and 50 had backtracked into the starting area. `map_rank` is a property of the
  stored state rather than of the trajectory, so it separates *has been far* from *is far*.
* **`hp_frac`** — see §4d.
* **`visits`** — the only term that ratchets *within* a map. Every other term is constant
  across a map, so in a maze the size of Viridian Forest nothing preferred the cells
  nearest the far exit. `visits` counts how often the world was reached at that cell,
  unlike `chosen`, which counts how often the sampler picked it. Measured in the forest:
  85.9 visits for entrance cells against 25.5 deep inside, corr(y, visits) = +0.73. Rarely
  reached means hard to reach, which in a maze means further in. It enters as a log rather
  than through the novelty term's 1/√, which would compress that 3.4× difference into a
  0.09 score gap — invisible under the softmax.
* **`chosen`** — the standard count-based bonus (Strehl & Littman 2008), stopping the
  archive collapsing onto one deep dead end.

Eviction never removes the unique deepest cell, which is what makes the frontier monotone
and hence makes assumption (A2) of the proof a theorem rather than a hope.

### 4b. Reading RAM that is mid-write

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

### 4c. `map_rank` is undefined for maps no milestone names, and that stalled the run for 38M steps

`MAP_RANK` is derived only from maps that some milestone predicate mentions. Connector maps
— gates, tunnels, the insides of buildings — are named by nobody and rank **−1**.

Viridian Forest North Gate (map 47) is exactly such a connector, and it is the *only* exit
from the forest. Measured: the three archived north-gate cells scored 12.6–12.9 against
17.8–18.1 for ordinary forest cells, with `chosen = 0` for all three across three hours.
**The states furthest along the path scored lowest**, so all 1119 restores went back into
the forest. Route 2 compounded it: it spans both sides of the forest and keeps rank 9 from
"On Route 2" (`map_rank` takes the *earliest* milestone naming a map), so emerging north
actually *lowered* a cell's rank below the forest it had just escaped.

The fix is a milestone rung, `forest_north_gate`, inserted between `viridian_forest` and
`pewter`. It is budget-neutral — its 400 expert steps come *out of* Pewter's 2500 rather
than being added, because `PROOF.md` §7 substitutes `TOTAL_EXPERT_STEPS` and the constant
must not drift silently. Gate cells then scored 23.6–23.9 against 21.0–21.9 for the best
forest cells, and the milestone went 11 → 13 within 66k steps after 38M stuck.

The general rule: **when the agent stalls at a map boundary, check `map_rank` of the
connector maps before suspecting the policy or the terrain.** A prior investigation of this
same stall concluded a trainer sealed the corridor; that was wrong (the run's own novelty
memory showed 716 of 718 open forest tiles visited, including both gate tiles), and the
terrain hypothesis cost two debugging rounds.

### 4d. Health filters the candidate set; it cannot be a score term

At 79.8M steps the run had been stalled for 20.8M. Sampling `FrontierArchive.sample()`
directly showed why: **75% of restores began below 0.3 HP and 55% began essentially dead**,
with 47% landing in Pewter City where not one of 24 cells was above 0.33 HP. An episode
that starts one hit from a blackout cannot fight, cannot level, and cannot survive the walk
to a Pokémon Center. No reward shaping could have rescued those episodes.

`hp_weight = 2.0` was never going to matter: it contributes at most 2.0 against a
milestone-plus-`map_rank` spread of ~25. But **raising it does not work either**, and the
reason is the important part — under a softmax, selection weight depends on cell *count* as
much as on score. Measured: a penalty of 4.0 let the 20 healthy post-blackout Route 1 cells
outweigh the 2 healthy north-gate cells and take 66% of restores; a penalty of 2.0 left the
24 dying Pewter cells still winning. Either way the depth ordering is collateral damage.

`FrontierArchive.viable()` filters the candidate set instead, which leaves depth ordering
untouched and needs no tuning: a cell is viable if it is above `frontier_min_hp`, or has an
empty party (the opening state reads 0.0 HP with nothing to heal — `level_sum == 0`
identifies these exactly, verified 76/76 against party size), or sits on a Pokémon Center
map. Result: 100% of restores viable.

The Pokécenter exemption is not a detail. The first version of the filter excluded 7 of the
9 archived Pokémon Center cells (every Pewter one, ≤0.25 HP) — precisely the states where
healing can be practised, six steps and an A press away — and so cancelled out a different
fix made hours earlier. **After adding any archive filter, check what it removes that
another fix was relying on.**

### 4e. What the archive accumulates, eviction deletes first

The per-level cap ranks victims by `map_rank`. Two classes of cell lose under that ranking
for reasons that have nothing to do with their value:

* **Utility maps.** No milestone names a shop or a Pokémon Center, so every one ranks −1
  and dies before anything on the critical path. Measured: every Viridian Mart cell had
  been deleted, leaving no archived state inside the only building that sells Poké Balls —
  while a reward term for acquiring balls sat waiting to fire.
  `UTILITY_MAPS` (Pokécenters ∪ Marts) plus `utility_cells_per_map = 4` shields a bounded
  number of each.
* **Strong parties.** The strongest party is usually *behind* the frontier, because it got
  strong grinding somewhere safe. The trim triggered by reaching milestone 14 took the
  archive's best party from 722 XP (level 10) back to 327 (level 8).
  `strongest_cells_kept = 8` shields the top cells by experience.

The asymmetry that justifies both: **coverage is recoverable by revisiting a map; strength
is not.** Whenever the archive gains a new thing it accumulates, the eviction ranking needs
checking too, not just the insert test.

### 4f. Ratcheting on the right granularity

`insert`'s `better` test originally compared `level_sum`, which moves only in whole levels.
A level costs several wins (level 8 → 9 is ~93 XP for a medium-slow species; a wild win
pays 20–30), so a state that had gained 40 XP compared *equal* and never replaced the
stored blob. Every episode restored to the same XP and threw the partial progress away.
Measured: six sampled cells all held **exactly 327 XP** after 82M steps — not one point of
experience had ever been banked, and levelling was therefore impossible unless a single
episode produced a whole level from a cold start.

Three separate defects had to be cleared before a single point of XP survived an episode,
and they are worth listing because each looked like the fix was already in:

1. **The ratchet was too coarse** — fixed by storing `Cell.exp` (3-byte big-endian at party
   struct offset `0x0E`) and adding `exp > existing.exp` to the `better` test.
2. **The snapshot never updated.** `_worker` took a save state only when the *cell key* was
   new — once per key per episode. `progress_key` hashes badges, dex, party size and story
   flags, and winning a wild battle changes none of them, so the stored state stayed as it
   was around step 2 and every point of experience earned afterwards was discarded at
   episode end. Measured at 86.4M steps: `battle_won` firing in 145 of 146 metric rows
   while no cell had ever exceeded 327 XP. Now a snapshot is also taken when
   `party_exp_sum` rises or HP improves by >0.05.
3. **Selection never used what was banked.** Experience then froze at 400 XP, because the
   cells carrying it are post-blackout Pallet Town / Route 1 states at `map_rank` 1–5 while
   selection is depth-first. Measured: **99.5% of restores drew a 327 XP cell.** Fixed with
   a sampling *reservation* (`strength_prob`, `strength_pool`) rather than another score
   term — closing the measured 11-point `map_rank` gap by weight would need ~150 and would
   drown depth entirely. Note the draw inside the reservation must be **uniform over the
   pool, not by `score`**: scoring within the pool just re-runs the depth comparison, and
   the first attempt gave the strong cell 0% of restores.

The generalisable lesson: **when a reward keys on a coarse counter, check whether the
underlying quantity accumulates at a finer granularity than the archive preserves.**
`hp_frac`, `level_sum` and `exp` all had this shape. So did `hp_frac` in a fourth way —
`_pack_info` simply had no `hp_frac` key, so `info.get("hp_frac", 1.0)` took its default on
every insert and the recorded health of 374 live cells took exactly two values.

### 4g. The milestone number measures the archive, not the policy

`restore_prob = 0.85` means 85% of training episodes begin from an archived save state.
A reported `milestone N/63` therefore means *"a worker restored into an archived state and
tripped that predicate"* — **not** *"the agent can play from the opening to there"*.

Measured at 57.8M env steps, driving the checkpoint exactly as `pokewm.play` does, 3000
steps each:

| start | milestone | ended | tiles |
|---|---|---|---|
| cold (default `play`) | 1 → 5 | Oak's Lab | 79 |
| cold + seeded novelty memory | 1 → 5 | Oak's Lab | 80 |
| `--from-frontier` (Pewter, ms 13) | 13 → 13 | **Route 1** | 38 |

From Pewter the policy walked *backwards* to Route 1. The archive carries the progress; the
standalone policy is much weaker. This is a real property of Go-Explore-style training and
not a bug, but it means the headline milestone number must never be read as policy
competence, and `pokewm.play` will not reproduce it.

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

## 5. Reward design

Four classes, each with a different reason it cannot be farmed. `RewardConfig` is the
authority; this table tracks it.

| term | weight | class |
|---|---|---|
| story flag | +4.0 each | monotone potential |
| badge | +64.0 each | monotone potential |
| party level sum | +2.0 each | monotone potential |
| party member gained | +8.0 each | monotone potential |
| Pokédex owned / seen | +2.0 / +0.2 | monotone potential |
| money | +0.0002 | monotone potential |
| **Poké Balls held** | **±1.5, capped at 6** | **symmetric potential** (§5b) |
| party HP fraction | ±3.0 | symmetric potential |
| status cured / inflicted | ±3.0 | symmetric potential |
| map rank progress | ±0.5 | symmetric potential |
| battle won | +3.0, latched | event |
| damage dealt to opponent | +1.0 over a full HP bar | event |
| Pokémon Center heal | +2.0, once per owed trip | event |
| party faint | −5.0 | event |
| new map / new tile | +3.0 / +0.02 | summable novelty |
| battle stall | −0.02/step after 24 | bounded penalty, cap 7.5 |
| dither | −0.005 | bounded penalty |
| step cost | −0.0005 | constant |
| epistemic (JSD) | ×1.0 | bounded by ln 4 |
| LLM subgoal | +3.0, once per assignment | verification-gated |

The dense `new_tile` term is doing quiet but essential work: it is what makes the *reward*
horizon O(10) steps even though the *milestone* horizon is O(10³).

### 5b. Monotone is right for levels and wrong for anything spendable

The original design applied one rule to every progress term: pay `w · Δ` on the *maximum*
of a counter, so the term can only ever be earned once. That is correct for badges, levels,
party size and the Pokédex, which cannot decrease. It is silently broken for anything the
agent can spend.

Oak hands over 5 Poké Balls, and the agent threw them all. `max_balls` was then pinned at a
value the bag could never reach again — measured at 1 while all 607 archived cells carried
0 balls, so **buying a ball would have paid exactly nothing**, forever. The term had been
in the config the whole time and had never once fired.

Balls are now a **symmetric potential**: buying pays +1.5, throwing costs −1.5, capped at
6. Because it telescopes it is unfarmable in the Ng–Harada–Russell sense, and unlike the
monotone form it stays payable for the whole run. `reward/ball` fired within 300k steps of
the change. `hp_potential`, `status_potential` and `map_progress` have the same shape for
the same reason.

**The rule:** monotone credit for quantities that cannot decrease; a symmetric potential
for everything else. Applying the monotone form to a consumable does not merely
under-reward it — it makes the term permanently dead the first time the resource is spent.

### 5c. Price the alternatives against each other, not against zero

Three separate stalls in this project had one shape: two bad outcomes, where the reward
made the *worse* one cheaper. Each was invisible in aggregate metrics and obvious the
moment the alternatives were priced side by side.

**Fleeing dominated fighting.** The reward had `enemy_damage` (at most 1.0 total for taking
an opponent from full to zero) and `faint` (−5.0), but **nothing for winning**. Fleeing
pays 0 at no risk, so fleeing strictly dominated fighting, and the policy had correctly
learned to flee. Measured over 4000 steps from healthy restores: 40 battles entered, party
level sum 8 → 8, in-battle action mix 23% B against 15% A — cycling the battle menu, then
running. `reward/level` had never fired in the entire 81M-step run. Fixed with
`battle_won = 3.0`, latched when the opponent's HP hits 0 and cashed when the battle ends
(the enemy struct is already stale by then, so it must be latched, not read at the end).
Fleeing never sets the latch, which is the entire point — the two outcomes have to be
*distinguishable*. 3.0 makes fighting positive-EV once win probability passes ~0.5.
Verified: fired in 57 of 59 metric rows within 630k steps, after 82M steps of silence.

**Stalling dominated losing.** `battle_stall` was capped at `|faint| × 0.5 = 2.5`,
deliberately below the 5.0 wipe penalty so that losing on purpose would not become a cheap
exit. That produced the mirror failure: sitting in an unwinnable trainer battle — a level-14
Onix against a level-8 Charmander, which can be neither won nor fled — was cheaper than
losing it. Measured over the 530k steps after the agent first engaged Brock, `battle_stall`
went from 0/69 metric rows to 48/52 while `battle_won` went 57/69 → **0/52** and `faint`
56/69 → **0/52**. It was not losing the fight; it was sitting in it. The cap is now
`|faint| × 1.5 = 7.5`, so the ordering is win (+3.0) > lose (−5.0) > stall (−7.5): losing
at least ends the fight and hands back a healthy state. A battle that has exhausted its
stall charge also stops deferring episode truncation, since it has demonstrably stopped
progressing.

**Blacking out dominated playing** — the original instance, written up in `PROOF.md` §6.1b.

The diagnostic that finds these: **when a signal goes quiet, ask what the agent switched
*to*, and price the two against each other.** Running the live policy from `archive.sample()`
restores and logging the in-battle action histogram made the first one obvious in minutes;
it was completely invisible in aggregate metrics.

### 5d. A term can be unearnable in exactly the situation it exists for

`heal_visit` fired **0 times in 59M steps**. `reset()` set `_heal_trip_paid = True`
unconditionally, and the flag is only cleared by a *downward crossing* of
`heal_subgoal_hp` within an episode. But 85% of episodes restore from the archive, and
every deep archived cell was captured hurt — so an episode starting below the threshold
never crossed it, and the reward for healing was unavailable to precisely the states that
needed to heal.

The consequence chain is worth following, because it explains a symptom that looked
unrelated: never healing means `wLastBlackoutMap` stays **Pallet Town** in all 370 cells
(Gen 1 defaults it to the player's house until a Center heal *completes*), so every faint
anywhere in the game costs the entire journey back. Healing is *easy* when attempted —
6 steps toward the counter plus A-mashing took a Pewter cell from 0.25 HP to 1.00. It was
never a difficulty problem; it was a credit problem.

Fixed via `_heal_trip_owed(gs)`, evaluated at reset: a trip is owed when the restored party
is hurt or ailing, and never when the party is empty (the opening state reads 0.0 HP with
nothing to heal). The Pewter Pokécenter cells that previously paid 0.0 now pay 2.0.

**The check this suggests, for any conditionally-gated reward: enumerate the states the
term exists to reward, and confirm the gate can actually open from them.** Under archive
restarts, "what an episode starts as" is drawn from the archive, not from the game's
opening — so any gate keyed on a within-episode transition is suspect by default.

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
collector thread drives 12 headless PyBoy subprocesses while the main thread takes gradient
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

**Sustained over a multi-day run**, the 88.7M-step run reported in the README averaged
**455 env-steps/s at 0.84 updates/s** over 211k updates — more collection and fewer
updates than the 12-env benchmark row (330 / 1.45). The two were measured under different
conditions (long run with a full replay buffer and a live LLM proposer, versus a short cold
benchmark), so the difference is not attributed to any single cause here. **455 steps/s is
the figure to plan a long run with**; the benchmark table is for comparing configurations
against each other, not for extrapolating wall-clock.

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
    ram_map.py            WRAM decoding -> GameState -> 32-D symbolic vector
    maps.py               the 248-entry map id table from pret/pokered,
                          plus POKECENTER_MAPS / MART_MAPS / UTILITY_MAPS
    bootstrap.py          scripted traversal of the title/naming screens
    env.py                Gymnasium env: observation, reward, save states
    vec_env.py            forkserver subprocess workers (12 by default)
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
    milestones.py         the 64-milestone critical path + MAP_RANK
    stall.py              plateau detection over the metrics stream
    trainer.py            concurrent collector + learner, checkpointing
  train.py                training entry point
  play.py                 live GUI viewer
  diagnose.py             offline inspection of a run's archive and metrics
```
