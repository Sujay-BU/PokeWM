# Why this system reaches the Hall of Fame

This document proves what can be proved and is explicit about what cannot.

The headline result is **Lemma 2**: the frontier archive converts the probability of
completing Pokémon Red from a *product* of 63 per-milestone success probabilities into a
*sum* of their reciprocals. At a per-milestone success rate of 0.1 that is the difference
between 10⁻⁶³ and roughly 3×10³ episodes — **59.5 orders of magnitude**. Everything else in
this document is either supporting machinery for that lemma or an honest accounting of its
assumptions.

Two theorems follow. **Theorem 1** is unconditional given the structural assumptions and
shows completion happens almost surely; its rate is astronomically slow because it uses
only the policy's guaranteed exploration floor. **Theorem 2** is the practically relevant
one and gives a finite-sample bound in terms of a learned per-milestone success
probability. §7 substitutes measured constants and lands at *tens of hours to a few
days*, which is consistent with the only comparable published result (Gemini 2.5 Pro's
813-hour and 406-hour Pokémon Blue completions, 2025).

---

## 1. Formal setup

### 1.1 The environment

Let the emulator define a deterministic controlled dynamical system

$$\mathcal{E} = (\mathcal{X}, \mathcal{A}, f), \qquad f : \mathcal{X}\times\mathcal{A}\to\mathcal{X}$$

where $\mathcal{X}$ is the set of full Game Boy machine states (work RAM, video RAM,
registers, the LCD and timer state) and $\mathcal{A}$ is the 7-element button set of
`pokewm.emulator.env.ACTIONS`.

**Determinism.** $f$ is a deterministic function. Pokémon Red contains no hardware
entropy source; its pseudo-randomness is derived from the divider register and frame
counters, which are themselves deterministic functions of the machine state. Two
consequences matter:

1. A saved state plus an action sequence reproduces a trajectory exactly. This is what
   makes "return to a previously reached state" an *exact* operation rather than an
   approximation, and it is the reason the Go-Explore construction of §3 applies without
   the return-policy machinery that stochastic domains require.
   Verified by `tests/test_env.py::TestDynamics::test_emulator_is_deterministic_given_the_same_actions`.
2. The agent nevertheless faces a POMDP, because the observation
   $o = O(x) \in \mathbb{R}^{5\times72\times80}\times\mathbb{R}^{32}\times\{0,1\}^{24}$
   discards most of $x$. The RSSM's recurrent state is the belief-state approximation.

### 1.2 The milestone chain

Define $L+1 = 64$ predicates $\varphi_0,\dots,\varphi_L$ on trajectory prefixes
(`pokewm/agent/milestones.py`), with $\varphi_0 \equiv \text{true}$ and $\varphi_L$ =
"the Hall of Fame map has been entered".

**Monotonicity (M).** Each $\varphi_i$ is monotone along a trajectory: if it holds at time
$t$ it holds at all $t' > t$. This holds by construction — every predicate is a threshold
on a quantity that only increases (a badge bit, a party count, a Pokédex count, membership
of the visited-map set). Verified by
`tests/test_milestones.py::TestMonotonicity`.

Define the **level** of a trajectory prefix $\tau$:

$$\ell(\tau) = \max\{\, i : \varphi_j(\tau)\ \text{holds for all } j \le i \,\} \in \{0,\dots,L\}.$$

By (M), $\ell$ is non-decreasing along any trajectory. Completing the game is exactly
$\ell = L$.

### 1.3 The algorithm as a stochastic process

An **episode** is: draw a launch state from the archive (or the game's opening state),
then run the policy for at most $T = 8192$ steps. Write $M_n$ for the archive's maximum
level after $n$ episodes. The run is in **phase $i$** while $M_n = i$.

---

## 2. The exploration floor

**Lemma 1 (positive support).** For every belief state $b$ and every action $a$, the
behaviour policy satisfies

$$\pi(a \mid b) \;\ge\; \varepsilon \;=\; \frac{u}{|\mathcal{A}|} \;=\; \frac{0.01}{7} \;\approx\; 1.43\times10^{-3}.$$

*Proof.* The actor's categorical distribution is passed through `unimix_logits` with
$u = 0.01$ (`pokewm/wm/actor_critic.py::Actor.logits`), which returns
$\log\big((1-u)\,p + u/|\mathcal{A}|\big)$ componentwise. Hence every probability is at
least $u/|\mathcal{A}|$ regardless of the logits. 

This is a property of the *implementation*, not an assumption: it holds at every point in
training, including after divergence, and is checked by
`tests/test_nets.py::TestUnimix::test_bounds_probabilities_away_from_zero`.

**Corollary 1.1.** Any fixed action sequence of length $H$ is executed with probability at
least $\varepsilon^{H} > 0$.

---

## 3. The archive: from product to sum

This is the central argument.

### 3.1 Assumptions

**(A1) Local reachability.** For each $i < L$ there exists $H_i < \infty$ such that from
the state stored in *some* archived cell at level $i$, an action sequence of length at
most $H_i$ produces level $\ge i+1$.

> *Justification.* Pokémon Red has no missable progression items: every key item, HM and
> gym leader remains obtainable indefinitely, and the player cannot dispose of HMs or of
> the last Pokémon in the party. The game is therefore completable from any legally
> reachable state. `MILESTONES[i].expert_steps` records a human-expert estimate of $H_i$;
> the largest is 6000 and the total is 125 900.
>
> Note the assumption is only about *some* cell at each level, not all of them. This is
> what the archive buys: a worker that walks into a locally unproductive state costs one
> episode, not the run, because the next episode relaunches from the archive.

**(A2) Frontier monotonicity.** $M_n$ is non-decreasing in $n$.

> *Proof, not assumption.* Cells are only ever inserted or deepened, never shallowed
> (`FrontierArchive.insert` replaces a blob only when `milestone` increases or ties), and
> `_evict_locked` refuses to evict a cell that is the unique deepest one. Verified by
> `tests/test_archive.py::TestEviction::test_frontier_never_regresses_under_pressure`,
> which hammers a capacity-6 archive with 500 random inserts and asserts the maximum
> never drops.

**(A3) Frontier selection.** Each episode is launched from a level-$M_n$ cell with
probability at least $\rho\,\sigma$, where $\rho = 0.85$ is `restore_prob` and $\sigma$ is
bounded in Lemma 2b.

### 3.2 Frontier selection probability

**Lemma 2b.** With `frontier_prob` $=\phi$, the selection rule draws its candidate set
from level $\ge M$ with probability $\phi$ and from the whole archive otherwise. Hence

$$\sigma \;=\; \Pr[\text{selected cell is at level } M] \;\ge\; \phi \;=\; 0.8,$$

independently of the score function, and unconditionally $\sigma \ge 1/C$ with $C = 4000$
the archive capacity.

*Proof.* Conditioned on the restricted draw, every candidate is at level $\ge M$, so the
selected cell is too; that event has probability $\phi$. The remaining mass is
non-negative. The unconditional bound follows because the largest of $C$ softmax weights
is at least $1/C$. 

#### Why the bound is stated this way (a correction)

An earlier version of this lemma bounded $\sigma$ through the softmax alone. With scores
$s(c) = \ell(c) + w/\sqrt{1+\text{chosen}(c)}$ and $w = 1$, a frontier cell has
$s \ge M$ and a cell at depth $k$ has $s \le M-k+w$, giving

$$\sigma \;\ge\; \frac{n_0}{\sum_{k\ge0} n_k e^{-(k-w)}} \;\ge\; \frac{n_0(e-1)}{n_{\max}e^{w+1}} \;\approx\; 0.233\,\frac{n_0}{n_{\max}}.$$

**That derivation is unsound for the score actually implemented.** The real score carries
two further terms — $\mu\,r(c)$ for map rank ($\mu = 1$, $r \le 63$) and $\beta = 6$ for
sitting on a target map — so a cell at depth $k$ can outscore a frontier cell by up to
$\mu R + \beta - k$. The depth term stops dominating and the geometric decay in $k$
disappears.

This was not a theoretical worry. Measured at 15.6M env steps: of the 24 archived Route 1
cells, 22 were one level stale, and they took $\approx 80\%$ of on-target restores even
though $n_0/n_{\max} \approx 1$. The stale level was pre-parcel-delivery, in which the old
man still blocks Viridian's north exit, so those episodes began in a world where the next
milestone was *unreachable* — $p_i = 0$, not merely small. The run held milestone 9 for
2.0M steps.

The lesson generalises beyond the bug: $\ell$ is not a "distance travelled" coordinate
that other terms may trade against. It indexes irreversible world state — gates opened,
key items held, NPCs moved — and A3 is a statement about *that*, so the selection rule has
to guarantee it structurally rather than through a weight comparison. Explicitly
restricting the candidate set does; a softmax over hand-weighted features does not.

$\phi < 1$ deliberately. A frontier level can be a dead end (a milestone predicate that is
satisfiable only through a state the archive never stored), and the $1-\phi$ unrestricted
mass is what lets the run abandon it. The ratio $n_0/n_{\max}$ is still logged as
`archive/frontier_frac`, now as a diagnostic rather than a term in the bound.

### 3.3 The main lemma

**Lemma 2 (exponential-to-linear reduction).** Suppose that from a level-$i$ frontier cell
a single episode attains level $\ge i+1$ with probability at least $p_i > 0$. Then:

**(a) Without archive resets** — every episode starts from the game's opening state — the
probability that a single episode completes the game is at most $\prod_{i=0}^{L-1} p_i$,
so the expected number of episodes is at least $\big(\prod_i p_i\big)^{-1}$. With
$p_i \equiv p$ this is $p^{-L}$.

**(b) With archive resets**, the expected number of episodes to completion satisfies

$$\mathbb{E}[N] \;\le\; \sum_{i=0}^{L-1}\frac{1}{\rho\,\sigma\,p_i} \;\le\; \frac{L}{\rho\,\sigma\,p_{\min}}.$$

*Proof.* (a) The milestone predicates are ordered and monotone, so within one episode
level $L$ requires passing through every intermediate level; the episode must therefore
succeed at all $L$ transitions, and its success probability is at most the product.

(b) By (A2) the process visits phases $0,1,\dots$ in order and never returns. Condition on
being in phase $i$. Each episode independently launches from a level-$i$ cell with
probability $\ge \rho\sigma$ (A3) and, given that, attains level $\ge i+1$ with probability
$\ge p_i$. Successive episodes are conditionally independent given the archive contents,
so the number of episodes spent in phase $i$ is stochastically dominated by a geometric
random variable with parameter $\rho\sigma p_i$, whose mean is $(\rho\sigma p_i)^{-1}$.
Phases are disjoint in time, so expectations add. 

**The point.** With $p = 0.1$ and $L = 63$:

| | expected episodes |
|---|---|
| no archive (Lemma 2a) | $10^{63}$ |
| with archive (Lemma 2b), $\rho\sigma = 0.2$ | $\approx 3.2\times10^{3}$ |

The archive does not make the agent smarter. It removes the requirement that the agent be
lucky 63 times *in a row*, which is the only reason the unassisted problem is hopeless.
This is the Go-Explore insight (Ecoffet et al., *Nature* 590, 2021) specialised to a
deterministic emulator, where "return to a cell" is a save-state load and therefore exact.

---

## 4. Theorem 1 — almost-sure completion

**Theorem 1.** Under (A1) and (A2), with the policy of Lemma 1 and $T \ge \max_i H_i$,

$$\Pr[\text{Hall of Fame reached within } N \text{ episodes}] \xrightarrow[N\to\infty]{} 1,$$

and the game is completed almost surely.

*Proof.* Fix phase $i$. By (A1) there is an action sequence of length $H_i \le T$ from some
archived level-$i$ cell that attains level $i+1$. That cell is selected with probability at
least $\rho/C > 0$ (Lemma 2b, unconditional form), and by Corollary 1.1 the policy emits
that exact sequence with probability at least $\varepsilon^{H_i} > 0$. Hence each episode
leaves phase $i$ with probability at least

$$q_i \;=\; \frac{\rho}{C}\,\varepsilon^{H_i} \;>\; 0 .$$

The number of episodes spent in phase $i$ is dominated by $\mathrm{Geom}(q_i)$, which is
finite almost surely. By (A2) the phases are traversed in order and never revisited, so the
total episode count is a sum of $L$ almost-surely finite random variables and is itself
almost surely finite. Equivalently, $\sum_n \Pr[\text{fail episode } n \mid \text{phase } i]
= \infty$ and the second Borel–Cantelli lemma applies to the independent trials in each
phase. 

**What Theorem 1 is and is not.** It establishes that the algorithm is not *structurally*
incapable of finishing: there is no state from which it is stuck, and no positive
probability of permanent failure. Its rate is worthless — $\varepsilon^{H_i}$ for
$H_i = 2000$ is about $10^{-5680}$. Theorem 1 says the floor is above zero. Theorem 2 says
what actually happens.

---

## 5. Theorem 2 — finite-sample bound

Theorem 1 used only the exploration floor. The learned components — world model, shaped
reward, imagination actor-critic, LLM subgoals — exist to replace $\varepsilon^{H_i}$ with
something usable.

**(A4) Learnability.** There is $p_{\min} > 0$ such that, after the world model has been
trained on data from phase $i$, an episode launched from a level-$i$ frontier cell attains
level $i+1$ with probability at least $p_{\min}$.

**Theorem 2.** Under (A1)–(A4), after $N$ episodes,

$$\Pr[\text{not completed}] \;\le\; L\,\exp\!\left(-\frac{N\,\rho\,\sigma\,p_{\min}}{L}\right),$$

so achieving failure probability at most $\delta$ requires

$$N \;\ge\; \frac{L}{\rho\,\sigma\,p_{\min}}\,\ln\frac{L}{\delta}.$$

*Proof.* Split the budget evenly, $N_i = N/L$ episodes per phase. Phase $i$ is not left
within $N_i$ episodes with probability at most $(1-\rho\sigma p_i)^{N_i} \le
e^{-N_i\rho\sigma p_{\min}}$. Union bound over the $L$ phases. 

The bound is linear in $L$ and in $1/p_{\min}$, and only logarithmic in $1/\delta$. That
shape — rather than the constants — is the design goal the architecture was built to hit.

---

## 6. What the reward signal does to the optimum

Theorem 2 assumes learning helps. This section accounts for the shaping used to *make* it
help: four of the five shaped classes provably leave the optimal policy set unchanged, and
the fifth does not. An earlier edition of this document was titled *"the reward signal does
not move the optimum"* and that claim is no longer true — §6.1c is the correction, and it
is a deliberate design decision rather than an oversight.

The per-step reward decomposes as

$$r_t \;=\; \underbrace{r^{\text{prog}}_t}_{\text{potential-based}} \;+\; \underbrace{r^{\text{evt}}_t}_{\text{repeatable events}} \;+\; \underbrace{r^{\text{nov}}_t}_{\text{first-visit novelty}} \;+\; \underbrace{r^{\text{epi}}_t}_{\text{model disagreement}} \;+\; \underbrace{r^{\text{sub}}_t}_{\text{LLM subgoal}} \;+\; \underbrace{r^{\text{step}}}_{\text{constant cost}} .$$

Four of these five shaped classes provably cannot move the optimum. The fifth,
$r^{\text{evt}}$, **can**, and §6.1c states by how much rather than pretending otherwise.

### 6.1 Progress terms are potential-based

$r^{\text{prog}}$ pays $w_k \Delta N_k$ on a set of state functions $N_k$. Within an
episode each $N_k$ is a function of state, so with

$$\Phi(s) \;=\; \sum_k w_k N_k(s)$$

we have exactly $r^{\text{prog}}_t = \Phi(s_{t+1}) - \Phi(s_t)$.

Two sub-classes differ in how $\Delta$ is taken, and the distinction is not cosmetic —
see §6.1a:

| sub-class | $\Delta$ taken against | members |
|---|---|---|
| **monotone** | the running maximum | story flags, badges, party level sum, party size, Pokédex owned/seen, money |
| **symmetric** | the previous step | party HP fraction, status conditions, Poké Balls held, map rank |

Ng, Harada & Russell (1999) show that shaping of the form $F = \gamma\Phi(s') - \Phi(s)$
leaves the optimal policy set unchanged. We use the $\gamma = 1$ form, so the discrepancy
is $(1-\gamma)\Phi(s')$, bounded by $(1-\gamma)\Phi_{\max}$ with $\gamma = 0.997$ and

$$\Phi_{\max} = \underbrace{4{\cdot}320}_{1280} + \underbrace{64{\cdot}8}_{512} + \underbrace{2{\cdot}600}_{1200} + \underbrace{8{\cdot}6}_{48} + \underbrace{2{\cdot}151}_{302} + \underbrace{0.2{\cdot}151}_{30} + \underbrace{2{\times}10^{-4}{\cdot}10^{6}}_{200} + \underbrace{1.5{\cdot}6}_{9} + \underbrace{3 + 3}_{6} + \underbrace{0.5{\cdot}63}_{32} \approx 3619 .$$

The *ordering* perturbation between two policies is at most
$(1-\gamma)\Phi_{\max} \approx 10.9$ per step of divergence. A badge is worth 64 and the
terminal milestone dominates, so the optimum ordering over policies that differ in whether
they finish the game is preserved — but note the margin has tightened from ~8.5× to ~5.9×
as terms were added. **$\Phi_{\max}$ is a budget, not a free parameter**: adding progress
weight indefinitely would eventually let per-step shaping discrepancy rival a badge.

### 6.1a Monotone credit is wrong for anything spendable

The monotone form pays on the running maximum, so a term can be earned once and never
again. That is correct precisely when the underlying quantity cannot decrease.

Applying it to a *consumable* does not merely under-reward the resource — it kills the term
permanently the first time the resource is spent. Measured: Oak hands over 5 Poké Balls,
the agent threw them all, and `max_balls` was then pinned at a value the bag could never
reach again — 1, while all 607 archived cells held 0 balls. Buying a ball would have paid
exactly nothing, forever, and `reward/ball` had never once fired.

The symmetric form is the fix, and it is the *more* conservative of the two with respect to
this section's argument: it telescopes exactly, so its contribution to any closed loop in
state space is identically zero. A cycle that spends and re-acquires a ball nets 0. This is
Ng–Harada–Russell in its original form; the monotone variant is the one that needs the
extra "can only increase" hypothesis. `reward/ball` fired within 300k steps of the change.

### 6.1b No per-step penalty may exceed the earning rate

A constraint the original design missed, and which cost a training run to find.

Let $c$ be the total per-step penalty (step cost plus dither) and $e$ the per-step
intrinsic earning rate. If $c > e$, then every state has negative expected value, and for
an episode of remaining length $H$ the agent can save $(c - e)H$ by reaching a terminal
state immediately. With $H \approx 6851$ and $c - e \approx 0.0133$ that was a **+89**
saving against a $-2$ blackout penalty, and the measured episode-termination rate duly
climbed from 0.23 to 0.41 as the policy discovered it.

Two independent guards now hold:

1. **Structural.** A party wipe is no longer terminal (`EnvConfig.terminate_on_wipe =
   False`). In the game a blackout teleports the player to a Pokecenter and play
   continues, so modelling it as an ending was simply wrong. With no reachable terminal
   state other than truncation, there is nothing to escape *to*, and the argument above
   cannot be run at all — no penalty tuning required.
2. **Numeric.** $c < e$ is asserted directly by
   `tests/test_env.py::TestRewards::test_per_step_penalties_cannot_exceed_the_intrinsic_earning_rate`,
   so the per-step baseline stays positive while exploring.

This is worth stating as a general rule for shaped long-horizon tasks: *a per-step penalty
larger than the per-step earning rate converts "end the episode" into the optimal policy,*
and the effect scales with the horizon — which is precisely the regime this project
operates in.

### 6.1c The event terms are *not* policy-invariant, and this is a deliberate trade

$r^{\text{evt}}$ — `battle_won` (+3.0), `enemy_damage` (+1.0 per opponent HP bar),
`heal_visit` (+2.0), `faint` (−5.0) — pays on *transitions*, not on a state function. It
does not telescope, and the positive members are repeatable without bound. **They can
therefore move the optimum, and this section quantifies the damage rather than denying it.**

**Why they exist.** The invariant reward produced a policy that would not fight. With
`enemy_damage` worth at most 1.0 for a full opponent bar and `faint` costing 5.0, fleeing —
which pays exactly 0 at no risk — *strictly dominated* fighting. Measured over 4000 steps
from healthy archive restores: 40 battles entered, party level sum 8 → 8, in-battle action
mix 23% B against 15% A. `reward/level` had never fired in 81M steps. Since levels gate the
entire mid-game, a reward that is policy-invariant and also incapable of producing a policy
that levels up is invariant around the wrong optimum.

**The exposure, quantified.** The grinding fixed point is: win wild battles forever. At
+3.0 per battle and ~50–100 steps per battle, that is 0.03–0.06 per step, worth
$0.06/(1-\gamma) = 20$ discounted at $\gamma = 0.997$. Against a badge at 64 that is not
negligible — it is roughly a third of a badge, permanently available. Three structural
facts bound it:

1. **Episodes truncate at $T = 8192$.** A grinding policy collects at most $\approx 3{\cdot}164 = 492$
   from `battle_won` in an episode, against $\Phi_{\max} \approx 3619$ of progress reward
   available on the critical path.
2. **The archive selects on $\ell$ and `map_rank`, not on return.** A worker that grinds
   produces cells that do not advance the frontier, so grinding does not propagate into
   the launch distribution the way progress does. This is the load-bearing mitigation:
   Lemma 2's phase argument depends on $M_n$, which `battle_won` cannot increase.
3. **The XP it buys is itself progress.** Unlike a true noisy-TV, the farmed quantity is
   the one that gates Brock, Mt. Moon and Misty. Grinding is the *intended* behaviour
   early; it only becomes pathological if it persists once levels are sufficient.

**Honest status.** (1) and (3) are bounds on the harm; (2) is the reason it does not
compound. None of them is a proof of policy invariance, and none is claimed to be. The
empirically observed failure has consistently been the *opposite* one — the agent
under-fights, and every measurement to date shows `battle_won` firing far below the
grinding rate — so the term is currently correcting a deficit rather than creating a
surplus. If a future run shows the agent grinding at a level well past what the next
milestone needs, the correct fix is to make the payout decay in party level (restoring a
bounded total, as in §6.2), not to remove it.

This is the one place in this document where a shaping term is known to perturb the
optimum. It is recorded in §9's failure-mode table.

### 6.2 Novelty terms are summable

$r^{\text{nov}}$ is *not* potential-based: `new_map` and `new_tile` use a run-level
first-visit rule, so revisiting a tile in a later episode pays nothing. It is instead a
count-based exploration bonus, and the relevant property is that its **total over the
entire run is bounded**:

$$\sum_{t=0}^{\infty} r^{\text{nov}}_t \;\le\; w_{\text{map}}\,|\mathcal{M}| + w_{\text{tile}}\,|\mathcal{C}| \;=\; 3{\cdot}248 + 0.02{\cdot}|\mathcal{C}| .$$

With at most $248$ maps of at most $\sim\!2^{14}$ tiles, this is under $10^{4}$ — finite,
and *independent of run length*. Hence the average novelty reward per step tends to $0$,
the bonus vanishes asymptotically, and the limiting objective is the unshaped one. This is
the standard optimism-under-summable-bonus condition (Strehl & Littman 2008, MBIE-EB).

### 6.3 The epistemic term is bounded and self-extinguishing

$r^{\text{epi}}$ is the Jensen–Shannon divergence of the 4-member ensemble, which satisfies
$0 \le \mathrm{JSD} \le \ln 4$ by construction
(`tests/test_wm.py::TestEnsemble::test_jsd_is_bounded_by_log_n`). At weight $1.0$ it is
therefore bounded by $\ln 4 \approx 1.386$ per step — still two orders of magnitude below
a badge (64), which is the bound that matters for §6.1's ordering argument, and checked by
`tests/test_env.py::TestRewards::test_epistemic_weight_stays_bounded_against_extrinsic_reward`.

The weight had to be raised from $0.1$: the first long run measured a typical
$\mathrm{JSD}$ of $0.019$, so the intrinsic drive was $0.00185$ per step against a step
cost of $0.002$. The two cancelled almost exactly, imagined return went *negative*, and
the agent's optimal imagined plan was to stand still — exploration stalled for 2.7M steps.
The step cost is now $0.0005$, an order of magnitude below the intrinsic term, so
exploration has a strictly positive gradient. Note this does not affect the argument
below: the bonus is still bounded and still decays to zero as the model fits. As the world model fits a region the
ensemble members converge and the term decays to zero there — it is an uncertainty
estimate, not a prediction error, so it does **not** persist on stochastic-but-learned
transitions (the noisy-TV failure mode; Pathak et al. 2017 → Plan2Explore, Sekar et al.
2020).

### 6.4 The LLM cannot move the optimum

Three properties, all enforced in code:

1. **Closed vocabulary.** `parse_subgoal` maps every possible model output into
   $\{0,\dots,23\}$; an unparseable or unknown answer becomes `MAIN_QUEST`, whose predicate
   is "a story flag advanced" — i.e. the extrinsic objective itself. A malformed response
   degrades to *no guidance*, never to *wrong guidance*.
   (`tests/test_subgoals.py::TestParsing`.)
2. **Verified payout.** The bonus fires only when a machine-checkable predicate over
   $(s_{t}, s_{t+1})$ holds, at most once per assignment
   (`tests/test_env.py::TestSubgoalBonus`). The LLM's *opinion* is never rewarded; only an
   actual state change is.
3. **No payout for regression.** Every predicate coincides with a non-decrease in some
   progress statistic, with the single deliberate exception of `USE_ITEM` (which fires on a
   bag-count decrease, because consuming a potion is progress).
   `tests/test_subgoals.py::TestNoRewardForRegression` asserts that a strictly-worse
   successor state pays out under no subgoal but that one.

Quantitatively, the proposer is single-flight with measured warm latency $\approx 7\,$s
while the emulator farm runs at $\approx 1.2\times10^3$ steps/s, so at most one bonus of
$3.0$ is payable per $\sim\!8\times10^3$ steps: an expected contribution below
$4\times10^{-4}$ per step, against a badge worth $64$. **The LLM is a proposal
distribution over exploration, not a term that can dominate the return.**

---

## 7. Substituting measured constants

| symbol | meaning | value | source |
|---|---|---|---|
| $L$ | milestone transitions | 63 | `NUM_MILESTONES - 1` |
| $\varepsilon$ | action-probability floor | $1.43\times10^{-3}$ | unimix $0.01/7$ |
| $\rho$ | archive restore probability | 0.85 | `ArchiveConfig.restore_prob` |
| $\sigma$ | frontier selection probability | $\ge 0.8$ | Lemma 2b (`frontier_prob`) |
| $T$ | steps per episode | 8192 | `EnvConfig.max_episode_steps` |
| $\sum_i H_i$ | expert steps, whole game | 125 900 | `TOTAL_EXPERT_STEPS` |
| — | measured throughput | **455 env steps/s** | sustained over 88.7M steps, 12 workers + concurrent learner |

With $\rho\sigma \ge 0.85 \times 0.8 = 0.68$ and $\delta = 0.05$:

$$N \;\ge\; \frac{63}{0.68\,p_{\min}}\,\ln\frac{63}{0.05} \;=\; \frac{661}{p_{\min}} \ \text{episodes}.$$

| $p_{\min}$ | episodes | env steps (at $T=8192$) | wall-clock at 455 steps/s |
|---|---|---|---|
| 0.30 | 2 205 | $1.8\times10^{7}$ | ~11 hours |
| 0.10 | 6 614 | $5.4\times10^{7}$ | ~33 hours |
| 0.05 | 13 228 | $1.1\times10^{8}$ | ~2.8 days |
| 0.01 | 66 140 | $5.4\times10^{8}$ | ~14 days |

Two corrections since the previous edition, pulling in opposite directions. $L$ rose from
45 to 63 as the milestone chain was refined (including the `forest_north_gate` rung added
to fix the connector-map stall of ARCHITECTURE §4c), which costs a factor of 1.47 in the
bound. And the throughput constant fell from an optimistic 1200 steps/s — a figure taken
from emulators running *without* a concurrent learner — to the 455 steps/s actually
sustained across an 88.7M-step run. **The honest table is roughly 3.7× slower than the one
it replaces**, and the earlier version should not be cited.

$p_{\min}$ — the probability a single frontier launch advances a milestone — remains the
term that decides the run and the one this document cannot derive from first principles.

This is otherwise a *pessimistic* reading: it charges every phase a full $T$-step episode,
whereas most milestones are reached in a few hundred steps and the run ends the episode
early on termination. It lands in the same order of magnitude as the only comparable
published completion — Gemini 2.5 Pro's 813-hour first run and 406-hour second run on
Pokémon Blue (2025) — which is the sanity check that matters.

**"Most likely reaches the goal" is a statement about
days to weeks of compute, not hours.** A 24-hour run is expected to clear part of the
early chain, not to finish the game. The system is built to be resumed, and §7 is the
argument for why resuming it converges.

### 7b. What the longest run to date actually did

The bound above is about $p_{\min}$ holding *uniformly*. The measured run shows what
happens when it does not — which is the practically important failure mode, and the
justification for the plateau-monitoring loop the project runs alongside training.

| | |
|---|---|
| env steps | 88 660 184 |
| gradient updates | 211 083 |
| best milestone | **15 / 63** — *Fighting in Pewter Gym* |
| badges | 0 |
| archive | 477 cells over 20 maps, max milestone 14 |

Progress was not rate-limited by $p_{\min}$ being uniformly small. It was rate-limited by a
handful of milestones where $p_i$ was **exactly zero** for structural reasons that had
nothing to do with the policy — a connector map ranked $-1$ so the archive never restored
to the forest exit (38M steps); the archive never banking experience so levelling was
impossible (27M steps); no reward for winning a battle so fleeing dominated fighting.
Each is documented in ARCHITECTURE §4c–§5d. After each fix the milestone moved within
30k–250k steps.

The lesson for (A4): the assumption most likely to fail is not "the world model cannot fit
this phase". It is "$p_i = 0$ because of a defect in the archive or the reward, and the run
looks healthy while it happens" — losses falling, throughput logged, no crashes. Every one
of these was found by instrumenting the live run, never by the test suite, because each is
a property of a long run rather than of a function.

---

## 8. Where the model-based part enters

Assumption (A4) is the one that is empirical rather than structural. Three mechanisms
justify it, and each is the reason a specific component exists.

**8.1 Short reward horizon.** With $\gamma = 0.997$ the critic's effective horizon is
$1/(1-\gamma) \approx 333$ steps, while $H_i$ is often $\sim 2000$. The gap is bridged by
the dense terms: `new_tile` fires every few steps during exploration, so the *reward*
horizon the critic must span is $O(10)$ steps even though the *milestone* horizon is
$O(10^3)$. Sparse-milestone credit assignment is never required.

**8.2 Imagination amplifies data.** Each gradient step replays $32 \times 32 = 1024$ real
transitions and generates $256 \times 15 = 3840$ imagined policy-gradient samples. The
policy is optimised against the model, so the emulator's throughput bounds *model*
learning, not *policy* learning. The classical guarantee is the simulation lemma (Kearns &
Singh 2002): for a model $\hat{P}$ with total-variation error
$\|\hat{P}(\cdot|s,a) - P(\cdot|s,a)\|_1 \le \epsilon_m$,

$$\big|V^{\pi}_{\hat{M}} - V^{\pi}_{M}\big| \;\le\; \frac{2\gamma R_{\max}\,\epsilon_m}{(1-\gamma)^2},$$

so improving the policy in imagination improves it in reality up to a term that shrinks as
the world model fits. This is why the world model must be the core rather than an
accessory: it is the object whose accuracy bounds the policy improvement.

**8.3 Directed exploration.** Within a phase, the epistemic bonus of §6.3 drives the agent
toward transitions the ensemble disagrees about — which, having just been launched from a
frontier cell, is precisely the unexplored region beyond it. This is what makes $p_i$
behave like a constant rather than like $\varepsilon^{H_i}$.

---

## 9. Assumptions, restated as failure modes

| assumption | status | how it could fail |
|---|---|---|
| Determinism of $f$ | **Verified** in tests | Would break exact cell return; Go-Explore's stochastic variant would be needed. |
| (M) Milestone monotonicity | **Proved** by construction + tests | Only if the predicate table is edited carelessly. |
| (A2) Frontier monotonicity | **Proved** in code + stress test | Only via an eviction bug. |
| Lemma 1 exploration floor | **Proved**, holds always | None; it is structural. |
| (A1) Local reachability | **Assumed**, justified by game design | A true soft-lock, or an archived cell at level $i$ from which $i{+}1$ is unreachable *and* no better cell is ever found. |
| (A3) Frontier selection | **Proved** given the archive's contents | Degrades if `frontier_frac` collapses — logged every run. |
| Policy invariance of $r^{\text{evt}}$ | **False, knowingly** (§6.1c) | Already false: `battle_won` admits a grinding fixed point worth ~20 discounted against a badge at 64. Bounded by episode truncation and by the archive selecting on $\ell$ rather than return. |
| Archive preserves what it accumulates | **Verified by repair, not by construction** | Held wrong for `hp_frac`, `level_sum` and `exp` simultaneously; each silently made some $p_i = 0$. Guarded by tests now, but this is the class of defect that has cost this project the most steps. |
| **(A4) Learnability** | **Assumed; the load-bearing one** | If the world model fails to fit some phase, $p_i \to \varepsilon^{H_i}$ and Theorem 2 degenerates into Theorem 1. Empirically the observed $p_i = 0$ cases were *all* archive or reward defects, not model capacity — see §7b. |

(A4) is where a claim of certainty would be dishonest. Nothing in this document proves the
RSSM can model Silph Co.; it proves that *if* it learns each phase to any constant success
rate, the archive makes the total cost linear rather than exponential in the number of
phases. Empirically, published Pokémon Red RL agents reach Cerulean City and the first gym
with far weaker machinery (PokeRL 2026: ~50% first-battle win rate at 500k steps;
"Pokémon Red via Reinforcement Learning" 2025: through Mt. Moon and Brock), which is direct
evidence for (A4) over the early chain. The later chain is extrapolation.

---

## 10. Summary

1. **Lemma 1** — the policy's action distribution is floored at $\varepsilon = 1.4\times10^{-3}$
   by unimix. Structural, always true.
2. **Lemma 2** — the frontier archive turns $\prod_i p_i$ into $\sum_i 1/p_i$. This is the
   whole reason the problem is tractable, and it is worth **~59.5 orders of magnitude** at
   $p = 0.1$ and $L = 63$.
3. **Theorem 1** — completion happens almost surely. Unconditional, but with a useless rate.
4. **Theorem 2** — $N \ge \frac{L}{\rho\sigma p_{\min}}\ln\frac{L}{\delta}$ episodes give
   failure probability $\le \delta$: linear in the number of milestones, logarithmic in
   confidence.
5. **§6** — the shaped reward is potential-based on its progress terms (monotone *and*
   symmetric), summable on its novelty terms, bounded on its intrinsic terms, and
   verification-gated on its LLM term. **The event terms are the exception**: `battle_won`
   and its relatives are repeatable, do not telescope, and admit a grinding fixed point
   worth ~20 discounted against a badge at 64. §6.1c states why that trade was taken and
   what bounds it.
6. **§7** — with measured constants, completion is a multi-day-to-multi-week proposition at
   $p_{\min}\approx0.05$–$0.1$, matching the only comparable published result. This is
   3.7× more pessimistic than the previous edition of the table.
7. **§7b** — in practice, the binding constraint has not been $p_{\min}$ being small. It
   has been $p_i$ being **exactly zero** at a handful of milestones because of defects in
   the archive or the reward, each of which left every other metric looking healthy.

### References

- Ecoffet, Huizinga, Lehman, Stanley, Clune. *First return, then explore.* Nature 590 (2021).
- Hafner, Pasukonis, Ba, Lillicrap. *Mastering diverse domains through world models* (DreamerV3), 2023.
- Kearns & Singh. *Near-optimal reinforcement learning in polynomial time.* MLJ 49 (2002).
- Ng, Harada, Russell. *Policy invariance under reward transformations.* ICML 1999.
- Sekar et al. *Planning to Explore via Self-Supervised World Models* (Plan2Explore). ICML 2020.
- Strehl & Littman. *An analysis of model-based interval estimation.* JCSS 74 (2008).
- *Simulus: Combining Improvements in Sample-Efficient World Model Agents*, arXiv:2502.11537 (2025).
- *PokeRL: Reinforcement Learning for Pokémon Red*, arXiv:2604.10812 (2026).
- *Pokémon Red via Reinforcement Learning*, arXiv:2502.19920 (2025).
