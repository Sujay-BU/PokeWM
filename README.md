# PokeWM: a World Model that learns to play Pokémon Red

An autonomous agent whose core is a **learned world model**: a discrete-latent RSSM
(DreamerV3 lineage) trained from scratch on a real Game Boy emulator, with its policy
optimised entirely inside latent imagination. A local Ollama model (`qwen3:8b`) runs
asynchronously alongside as a subgoal proposer, and a Go-Explore frontier archive converts
the game's ~10⁵-step horizon into a chain of short-horizon problems.

Nothing is pretrained. The world model, actor and critic all start from random
initialisation and learn from emulator interaction only.

- **[docs/PROOF.md](docs/PROOF.md)**: the convergence argument, including the lemma that
  makes the problem tractable (the archive turns a *product* of 45 per-milestone success
  probabilities into a *sum* of their reciprocals, worth ~49 orders of magnitude), and an
  honest account of which assumption is load-bearing.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**: every design decision with its
  rationale and, where it was settled by measurement, the measurement.

---

## Quick start

```bash
conda env create -f environment.yml
conda activate pokewm
pip install --index-url https://download.pytorch.org/whl/cu128 torch
pip install -r requirements.txt

# Put the ROM at roms/Pokemon - Red Version (USA, Europe) (SGB Enhanced).gb

pytest
python -m pokewm.train --preset laptop --logdir runs/overnight
python -m pokewm.play  --logdir runs/overnight    # watch it in a real GUI window
```

The post-intro save state is **derived from the ROM on first run** (0.5 s) rather than
shipped, see `pokewm/emulator/bootstrap.py`.

---

## How it works

```
        ┌─ 8 headless PyBoy workers ─┐         ┌──────── GPU ────────┐
        │  emulate, decode RAM,       │  data   │ RSSM world model    │
        │  write frontier save states ├────────►│ (59 M params)       │
        └──────────┬──────────────────┘         │        │            │
                   │ actions                    │        ▼            │
                   └────────────────────────────┤  imagination        │
                                                │  actor-critic       │
        ┌─ qwen3:8b (CPU, async) ─┐  subgoals   │  15-step rollouts   │
        │  one in-flight request  ├────────────►│  in latent space    │
        └─────────────────────────┘             └─────────────────────┘
                   ▲                                      │
                   │ state summaries          ┌────────────▼───────────┐
                   └──────────────────────────┤ Go-Explore frontier    │
                                              │ archive (save states)  │
                                              └────────────────────────┘
```

**World model (the core).** A recurrent state-space model with 32×32 categorical latents.
The deterministic GRU path carries long-range context; the stochastic path absorbs what it
cannot predict. Policy learning happens *only* on rollouts through the model's prior, one
replay batch of 32×32 real steps yields 256×15 ≈ 3 840 imagined policy-gradient samples, so
emulator throughput bounds *model* learning rather than *policy* learning.

**Frontier archive.** Episodes relaunch from archived save states, keyed by irreversible
progress (badges, story flags, party) **plus milestone index and a coarse position
bucket**. Because the emulator is deterministic, "return to a previously reached state" is
exact. This is the component that makes the horizon tractable, and the position term is
load-bearing, not cosmetic: without it the archive freezes solid inside any phase that
sets no story flag. See ARCHITECTURE §4.

**LLM subgoal proposer.** `qwen3:8b` picks one subgoal from a closed 24-item vocabulary.
Each subgoal carries a machine-checkable predicate over the decoded RAM, so the bonus pays
only when the suggestion *actually happened*, a wrong or garbled suggestion pays nothing
and cannot move the optimum. The RL loop never blocks on it.

**Exploration.** Jensen–Shannon disagreement across a 4-member ensemble of latent
predictors, epistemic uncertainty, not prediction error, so wild-encounter RNG does not
farm it (the noisy-TV failure mode).

---

## Measured on the development machine

RTX 4050 Laptop (6 GB), 16-core CPU, 16 GB RAM.

| | |
|---|---|
| World model | 59.4 M params, 3.70 GiB peak VRAM |
| Learner in isolation | 2.30 updates/s (2 360 replayed steps/s) |
| Emulators in isolation | 657 steps/s single worker; 1 874 steps/s across 8 |
| **End-to-end (12 envs, both running)** | **~330 env-steps/s at ~1.4 updates/s** |
| LLM proposer | ~7 s warm latency, CPU-only, never blocks training |
| Test suite | 309 passing, ~4.5 min |

End-to-end is well below either component in isolation: on one small GPU the collector's
policy forward queues behind the learner's kernels. Throughput scales nearly linearly with
worker count (213 / 330 / 438 steps/s at 8 / 12 / 16 envs); 12 is the point where memory
still leaves headroom on 16 GB. See ARCHITECTURE §8.

---

## Status and honest expectations

**What is verified.** The full stack runs end to end on the GPU: emulator → replay →
world model → imagination actor-critic → archive → checkpoint → resume → live GUI
playback. **309 tests pass**, including integration tests that boot the real ROM, train,
checkpoint and resume.


**What the proof actually claims.** Not that this finishes the game in a session.
`docs/PROOF.md` §7 substitutes measured constants and lands at *tens of hours to several
days* of continuous training for a high-probability completion, depending on the
per-milestone success rate, consistent with the only comparable published result (Gemini
2.5 Pro's 813-hour and 406-hour Pokémon Blue completions in 2025). A 24-hour run should be
expected to clear the early-to-middle chain (Brock → Mt. Moon → Misty), not the Elite Four.
The system is built to be resumed, and the proof is the argument for why resuming it
converges.

---

## Running the long job

Training is fully resumable, re-invoking with the same `--logdir` restores model,
optimiser, frontier archive and counters.

```bash
# supervised: auto-restarts from the last checkpoint if the trainer dies
# (OOM kill, GPU fault, transient CUDA error). This is the recommended mode
# for a multi-day run.
scripts/train.sh supervise --preset laptop --envs 12

# or unsupervised
scripts/train.sh start --preset laptop --envs 12
scripts/train.sh status
scripts/train.sh stop            # SIGINT -> checkpoints, then exits (~6 s)
scripts/train.sh start --preset laptop    # resumes where it left off

# or drive it directly
python -m pokewm.train --preset laptop --logdir runs/overnight
tail -f runs/overnight/train.log
cat runs/overnight/events.jsonl        # one line per milestone / badge

# watch it play, live, in a GUI window, safe to run next to training
python -m pokewm.play --logdir runs/overnight --from-frontier
```

The viewer renders every emulator frame (training renders only one frame per action, which
is right for throughput but shows ~2.5 fps in a window and reads as flicker). It prints
`N tiles/200 steps` so a policy pinned against a wall is visibly distinct from a frozen
emulator, and `--restart-if-stuck` (default 400 steps) jumps to another archived cell when
a mid-training policy wedges itself, so the window keeps showing something.

Checkpoints are written every 20 000 env steps **or** every 5 minutes, whichever comes
first, and both SIGINT and SIGTERM checkpoint before exiting.

Useful flags: `--preset cpu` (no-GPU fallback, ~7 M params), `--no-llm` (disable the
proposer), `--no-archive` (ablate Go-Explore), `--llm-vision` (use `qwen3-vl:8b`; slower,
see ARCHITECTURE §7), `--fresh` (ignore checkpoint), `--envs N`, `--replay-ratio R`.

### If the GPU faults

Symptom: `torch.cuda.is_available()` is `False` while `nvidia-smi` shows `ERR!` or
`[GPU requires reset]`. Recover with:

```bash
sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm
# if that is not enough:
sudo nvidia-smi -r          # or reboot
```

Training then resumes from the last checkpoint with no loss beyond the last
`checkpoint_every` window. The code also runs on CPU (`--device cpu`), roughly 15× slower.

---

## Tests

```bash
pytest                        # everything (~2 min)
pytest -m "not emulator"      # pure logic, no ROM needed (~10 s)
pytest -m llm                 # requires a running Ollama daemon
```

| file | covers |
|---|---|
| `test_ram_map.py` | WRAM offsets re-derived from known addresses, BCD money, symbolic encoding |
| `test_milestones.py` | chain structure, predicate monotonicity, tracker |
| `test_subgoals.py` | vocabulary, tolerant parsing, verification predicates, no-reward-for-regression |
| `test_nets.py` | symlog/symexp, two-hot expectation, λ-returns vs closed form, KL free bits |
| `test_wm.py` | RSSM reset semantics, gradient coverage, JSD bounds, actor-critic |
| `test_replay.py` | ring wrap, frame-stack reconstruction across episode boundaries, priorities |
| `test_archive.py` | insertion, eviction, **frontier monotonicity under pressure**, persistence |
| `test_env.py` | ROM hash, bootstrap, determinism, save states, reward first-visit credit |
| `test_llm.py` | JSON extraction, offline degradation, scheduling, live Ollama |
| `test_trainer.py` | end-to-end run, checkpoint/resume, replay ratio |

---

## Requirements

- A legally obtained Pokémon Red (USA/Europe, SGB Enhanced) ROM in `roms/`
- NVIDIA GPU with ≥4 GB VRAM (or CPU, slower)
- [Ollama](https://ollama.com) with `ollama pull qwen3:8b`, optional; the agent degrades
  gracefully to a scripted fallback if the daemon is unreachable
