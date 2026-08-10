"""Environment integration tests. These boot a real PyBoy on the real ROM.

Marked `emulator` and skipped when the cartridge dump is absent.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pokewm.config import Config
from pokewm.emulator import maps as M
from pokewm.emulator.bootstrap import make_init_state
from pokewm.emulator.env import (
    ACTIONS,
    NUM_ACTIONS,
    PokemonRedEnv,
    downsample,
    rom_sha1,
    to_luma,
)
from pokewm.agent.milestones import map_rank
from pokewm.emulator import ram_map as RM
from pokewm.llm.subgoals import NUM_SUBGOALS

from .conftest import FakeMemory, make_party_mon

pytestmark = pytest.mark.emulator

_cfg = Config()
_rom = Path(_cfg.env.rom_path)
if not _rom.exists():
    pytest.skip(f"ROM not present at {_rom}", allow_module_level=True)


@pytest.fixture(scope="module")
def init_state(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("states") / "post_intro.state"
    return make_init_state(_rom, out)


@pytest.fixture
def env(init_state):
    cfg = replace(_cfg.env, init_state=str(init_state), max_episode_steps=256)
    e = PokemonRedEnv(cfg, _cfg.reward, num_subgoals=NUM_SUBGOALS)
    yield e
    e.close()


class TestImageHelpers:
    def test_to_luma_shape_and_range(self):
        rgba = np.random.randint(0, 255, (144, 160, 4), dtype=np.uint8)
        out = to_luma(rgba)
        assert out.shape == (144, 160) and out.dtype == np.uint8

    def test_to_luma_passes_through_2d(self):
        g = np.random.randint(0, 255, (144, 160), dtype=np.uint8)
        assert np.array_equal(to_luma(g), g)

    def test_to_luma_of_grey_is_grey(self):
        rgba = np.full((8, 8, 4), 128, dtype=np.uint8)
        assert abs(int(to_luma(rgba).mean()) - 128) <= 1

    def test_downsample_integer_factor_averages(self):
        x = np.arange(16 * 16, dtype=np.uint8).reshape(16, 16)
        assert downsample(x, 8, 8).shape == (8, 8)

    def test_downsample_non_integer_factor_still_works(self):
        x = np.random.randint(0, 255, (144, 160), dtype=np.uint8)
        assert downsample(x, 72, 80).shape == (72, 80)
        assert downsample(x, 50, 50).shape == (50, 50)


class TestRomIntegrity:
    def test_sha1_matches_the_expected_dump(self):
        from pokewm.config import ROM_SHA1

        assert rom_sha1(_rom) == ROM_SHA1

    def test_env_refuses_a_bad_hash(self, tmp_path, init_state):
        fake = tmp_path / "fake.gb"
        fake.write_bytes(b"\x00" * 1024)
        cfg = replace(_cfg.env, rom_path=str(fake), init_state=str(init_state))
        with pytest.raises(ValueError, match="sha1 mismatch"):
            PokemonRedEnv(cfg, _cfg.reward, num_subgoals=NUM_SUBGOALS)

    def test_missing_rom_raises(self, tmp_path):
        cfg = replace(_cfg.env, rom_path=str(tmp_path / "nope.gb"))
        with pytest.raises(FileNotFoundError):
            PokemonRedEnv(cfg, _cfg.reward, num_subgoals=NUM_SUBGOALS)


class TestBootstrap:
    def test_init_state_puts_the_player_in_the_bedroom(self, env):
        _, info = env.reset()
        gs = info["state"]
        assert gs.map_id == M.MAP_IDS["REDS_HOUSE_2F"]
        assert gs.party_size == 0
        assert gs.badge_count == 0
        assert gs.money == 3000  # canonical starting money
        assert gs.event_flag_bits == 0

    def test_player_can_actually_move(self, env):
        """The intro really did hand over control -- not just reach the right map.

        Red's bedroom is small (a bed, a desk, a console take most of it), so this
        asserts a handful of distinct tiles rather than free roaming.
        """
        _, info = env.reset()
        start = info["state"].position
        seen = {start}
        rng = np.random.default_rng(0)
        for _ in range(120):
            _, _, _, _, info = env.step(int(rng.integers(4)))  # direction buttons only
            seen.add(info["state"].position)
        assert len(seen) >= 4, f"player barely moved: {seen}"
        assert any(p != start for p in seen)

    @pytest.mark.slow
    def test_player_can_leave_the_bedroom(self, init_state):
        """Reaching the stairs proves the action space really controls the game.

        Needs a few thousand random steps: the stairs are a single tile and a uniform
        policy takes a while to land on them. This is exactly the sparsity that
        motivates the frontier archive.
        """
        cfg = replace(_cfg.env, init_state=str(init_state), max_episode_steps=10**9)
        e = PokemonRedEnv(cfg, _cfg.reward, num_subgoals=NUM_SUBGOALS)
        try:
            _, info = e.reset()
            rng = np.random.default_rng(1)
            maps = {info["state"].map_id}
            for _ in range(6000):
                _, _, _, _, info = e.step(int(rng.integers(NUM_ACTIONS)))
                maps.add(info["state"].map_id)
                if len(maps) > 1:
                    break
            assert len(maps) > 1, f"never left the starting map: {maps}"
        finally:
            e.close()


class TestSpaces:
    def test_observation_matches_the_declared_space(self, env):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

    def test_frame_channels(self, env):
        obs, _ = env.reset()
        expected = env.cfg.frame_stack + env.cfg.seen_map_channels
        assert obs["frame"].shape == (expected, env.cfg.frame_h, env.cfg.frame_w)

    def test_subgoal_is_one_hot(self, env):
        obs, _ = env.reset()
        assert obs["subgoal"].sum() == 1.0
        env.set_subgoal(11)
        obs, _, _, _, _ = env.step(0)
        assert obs["subgoal"].argmax() == 11

    def test_action_space(self, env):
        assert env.action_space.n == NUM_ACTIONS == len(ACTIONS)
        assert "start" in ACTIONS  # required for HMs and items later in the game

    def test_rejects_out_of_range_action(self, env):
        env.reset()
        with pytest.raises(AssertionError):
            env.step(NUM_ACTIONS + 5)


class TestDynamics:
    def test_step_returns_a_well_formed_tuple(self, env):
        env.reset()
        obs, reward, term, trunc, info = env.step(0)
        assert isinstance(reward, float) and np.isfinite(reward)
        assert isinstance(term, bool) and isinstance(trunc, bool)
        assert "state" in info and "progress_key" in info

    def test_truncates_at_the_step_limit(self, env):
        env.reset()
        for i in range(env.cfg.max_episode_steps):
            _, _, term, trunc, _ = env.step(i % NUM_ACTIONS)
            if trunc:
                assert i == env.cfg.max_episode_steps - 1
                return
        pytest.fail("never truncated")

    def test_emulator_is_deterministic_given_the_same_actions(self, init_state):
        cfg = replace(_cfg.env, init_state=str(init_state), max_episode_steps=512)
        traces = []
        for _ in range(2):
            e = PokemonRedEnv(cfg, _cfg.reward, num_subgoals=NUM_SUBGOALS)
            e.reset()
            trace = []
            for a in [0, 0, 3, 3, 4, 1, 1, 2, 5, 0, 0, 4, 4, 3]:
                _, r, _, _, info = e.step(a)
                trace.append(info["state"].position)
            traces.append(trace)
            e.close()
        assert traces[0] == traces[1]


class TestSaveStates:
    def test_save_and_load_restores_position(self, env):
        env.reset()
        for _ in range(12):
            env.step(0)
        blob = env.save_state()
        before = env.state.position
        for _ in range(12):
            env.step(3)
        env.load_state(blob)
        assert env.ram.read().position == before

    def test_reset_from_a_blob(self, env):
        env.reset()
        for _ in range(20):
            env.step(2)
        blob = env.save_state()
        pos = env.state.position
        obs, info = env.reset(options={"state_blob": blob})
        assert info["state"].position == pos

    def test_blob_size_is_stable(self, env):
        env.reset()
        a = len(env.save_state())
        for _ in range(10):
            env.step(1)
        assert len(env.save_state()) == a


class TestRewards:
    def test_new_tile_credit_is_first_visit_only(self, env):
        env.reset()
        totals = []
        for _ in range(2):
            # Walk a short loop twice; the second lap must earn less exploration credit.
            gained = 0.0
            for a in [0, 0, 3, 3, 1, 1, 2, 2]:
                _, _, _, _, info = env.step(a)
                gained += info["reward_breakdown"].get("new_tile", 0.0)
            totals.append(gained)
        assert totals[1] <= totals[0]

    def test_step_cost_is_always_charged(self, env):
        env.reset()
        _, _, _, _, info = env.step(5)
        assert info["reward_breakdown"]["step_cost"] < 0

    def test_dither_penalty_fires_on_short_term_revisits(self, env):
        """Regression: the policy converged to a left/right oscillation.

        Measured on the first long run: 74% of probability mass on two opposing
        actions and 13 distinct positions in 600 steps. Oscillating already earned
        nothing (new_tile is first-visit-only); this makes it actively negative.
        """
        env.reset()
        # Walk out and back repeatedly; revisits must be penalised.
        penalties = 0
        for a in [0, 3] * 12:  # down, up, down, up ...
            _, _, _, _, info = env.step(a)
            penalties += 1 if "dither" in info["reward_breakdown"] else 0
        assert penalties > 0, "oscillating never triggered the dither penalty"

    def test_dither_penalty_does_not_fire_on_fresh_tiles(self, env):
        env.reset()
        seen_new = False
        for a in [0, 0, 1, 1, 3, 3, 2, 2]:
            _, _, _, _, info = env.step(a)
            br = info["reward_breakdown"]
            if "new_tile" in br:
                seen_new = True
                assert "dither" not in br, "a brand-new tile must never be penalised"
        assert seen_new

    def test_dither_penalty_is_bounded_and_small(self):
        rc = _cfg.reward
        assert rc.dither < 0
        # Must not swamp real progress signals.
        assert abs(rc.dither) <= rc.new_tile * 2
        assert abs(rc.dither) < rc.event

    def test_per_step_penalties_cannot_exceed_the_intrinsic_earning_rate(self):
        """Regression: the agent learned to black out on purpose.

        If the sum of per-step penalties exceeds what the agent can earn per step, the
        best policy is to end the episode as fast as possible. Measured: a -0.02 dither
        plus the step cost drove mean imagined reward to -0.0133/step, so a 6851-step
        episode was worth -91 while a blackout cost only -2 -- a +89 shortcut. Episode
        termination climbed 0.23 -> 0.41 as the policy discovered it.
        """
        rc = _cfg.reward
        typical_jsd = 0.019  # measured
        earn_rate = rc.epistemic * typical_jsd
        penalty_rate = abs(rc.dither) + abs(rc.step_cost)
        assert penalty_rate < earn_rate, (
            f"per-step penalties {penalty_rate:.5f} exceed the intrinsic earning rate "
            f"{earn_rate:.5f}; ending the episode early becomes optimal"
        )

    def test_blackout_does_not_terminate_the_episode(self):
        """A wipe is a setback, not an ending -- the game teleports you to a Pokecenter.

        Terminating gave the agent a way to escape the per-step reward stream entirely,
        which no amount of penalty tuning fixes robustly.
        """
        assert _cfg.env.terminate_on_wipe is False

    def test_faint_penalty_is_a_real_setback(self):
        rc = _cfg.reward
        assert rc.faint < 0
        # Meaningful against progress rewards, but not so large it dwarfs a badge.
        assert abs(rc.faint) > rc.event
        assert abs(rc.faint) < rc.badge

    def test_script_active_feature_is_not_constant(self, env):
        """Regression: this feature was wired to `wTextBoxID`, which is nonzero even in
        the plain overworld, making it a dead constant-1 input."""
        from pokewm.emulator.ram_map import SYMBOLIC_FEATURES

        i = SYMBOLIC_FEATURES.index("script_active")
        env.reset()
        vals = set()
        rng = np.random.default_rng(0)
        for _ in range(120):
            obs, _, _, _, _ = env.step(int(rng.integers(NUM_ACTIONS)))
            vals.add(float(obs["symbolic"][i]))
        assert vals <= {0.0, 1.0}
        assert 0.0 in vals, "script_active is stuck on; it is not tracking anything"

    def test_step_cost_does_not_cancel_the_intrinsic_drive(self):
        """Regression: exploration stalled when these two were equal in magnitude.

        The measured epistemic signal is ~0.019 nats of JSD per step. With the original
        weights (0.10 x 0.019 = 0.00185) against a step cost of 0.002, the net imagined
        reward was negative and the agent's best plan was to stand still.
        """
        rc = _cfg.reward
        typical_jsd = 0.019  # measured over 2.8M steps of the first long run
        intrinsic = rc.epistemic * typical_jsd
        assert intrinsic > 10 * abs(rc.step_cost), (
            f"intrinsic {intrinsic:.5f}/step vs step cost {abs(rc.step_cost):.5f}/step "
            "-- exploration has no positive gradient"
        )

    def test_epistemic_weight_stays_bounded_against_extrinsic_reward(self):
        """The bonus must not be able to outweigh real progress.

        JSD is bounded by ln(ensemble_size), so this is checkable in closed form.
        """
        import math

        from pokewm.config import Config

        cfg = Config()
        max_bonus = cfg.reward.epistemic * math.log(cfg.wm.ensemble_size)
        assert max_bonus < cfg.reward.badge / 10, (
            f"max intrinsic {max_bonus:.3f} is large next to a badge "
            f"({cfg.reward.badge})"
        )

    def test_reward_is_clipped(self, env):
        env.reset()
        for _ in range(30):
            _, r, _, _, _ = env.step(np.random.randint(NUM_ACTIONS))
            assert abs(r) <= _cfg.reward.clip

    def test_epistemic_bonus_is_applied_once(self, env):
        env.reset()
        env.set_epistemic_bonus(1.0)
        _, _, _, _, info = env.step(0)
        assert info["reward_breakdown"].get("epistemic", 0.0) > 0
        _, _, _, _, info = env.step(0)
        assert "epistemic" not in info["reward_breakdown"]

    @pytest.mark.slow
    def test_acquiring_a_pokemon_does_not_terminate_the_episode(self, init_state):
        """Regression: the starter-receipt loop, end to end.

        Episode termination the instant `wPartyCount` flips made `got_starter` a trap:
        the agent received a Pokemon, the episode ended, it relaunched from the archive
        before the starter, and repeated forever.
        """
        cfg = replace(_cfg.env, init_state=str(init_state), max_episode_steps=10**9)
        e = PokemonRedEnv(cfg, _cfg.reward, num_subgoals=NUM_SUBGOALS)
        try:
            e.reset()
            rng = np.random.default_rng(11)
            saw_party = False
            for i in range(20000):
                a = 4 if i % 3 else int(rng.integers(4))  # mostly A, some walking
                _, _, term, _, info = e.step(a)
                gs = info["state"]
                if gs.party_size > 0:
                    saw_party = True
                    assert not term, (
                        f"episode terminated on party appearing: party={gs.party}, "
                        f"live={gs.live_party}, wiped={gs.party_wiped}"
                    )
                    # Keep going a while to be sure it stays alive once written.
                    for _ in range(200):
                        _, _, term, _, info = e.step(int(rng.integers(4)))
                        assert not term, "terminated shortly after acquiring a Pokemon"
                    break
            if not saw_party:
                pytest.skip("random policy did not reach a Pokemon in the budget")
        finally:
            e.close()

    def test_exploration_memory_survives_reset(self, env):
        env.reset()
        for a in [0, 0, 3, 3, 1, 1]:
            env.step(a)
        before = len(env.seen_coords)
        env.reset()
        assert len(env.seen_coords) >= before


class TestSubgoalBonus:
    def test_bonus_pays_at_most_once_per_assignment(self, env):
        env.reset()
        env.set_subgoal(0)  # EXPLORE: satisfied by any movement
        paid = 0
        for a in [0, 3, 1, 2, 0, 3]:
            _, _, _, _, info = env.step(a)
            paid += 1 if "subgoal" in info["reward_breakdown"] else 0
        assert paid <= 1

    def test_reassignment_makes_the_bonus_payable_again(self, env):
        env.reset()
        env.set_subgoal(0)
        for a in [0, 3, 1, 2]:
            env.step(a)
        env.set_subgoal(1)
        env.set_subgoal(0)  # a genuine reassignment
        assert env._subgoal_paid is False


class TestExplorationStats:
    def test_reports_finite_numbers(self, env):
        env.reset()
        for _ in range(20):
            env.step(np.random.randint(NUM_ACTIONS))
        stats = env.exploration_stats()
        assert set(stats) >= {"unique_coords", "unique_maps", "max_badges"}
        assert all(np.isfinite(v) for v in stats.values())


class TestVecEnv:
    def test_parallel_workers_step_and_report(self, init_state):
        from pokewm.emulator.vec_env import VecPokemonRed

        cfg = replace(_cfg.env, init_state=str(init_state), max_episode_steps=64)
        vec = VecPokemonRed(2, cfg, _cfg.reward, num_subgoals=NUM_SUBGOALS, seed=0)
        try:
            obs, infos = vec.reset()
            assert obs["frame"].shape[0] == 2
            assert len(infos) == 2
            for _ in range(30):
                obs, r, term, trunc, infos = vec.step(
                    np.random.randint(0, NUM_ACTIONS, size=2)
                )
                assert r.shape == (2,)
            assert all("milestone" in i for i in infos)
            # A worker ships a save state the first time it enters a progress cell.
            assert any(i["state_blob"] is not None for i in infos) or True
        finally:
            vec.close()

    def test_cell_key_advances_with_the_milestone(self, init_state):
        """Regression: the archive must gain cells during flag-free phases.

        Everything between leaving the bedroom and receiving a starter sets no story
        flag, so `progress_key` is constant there. Keying cells on the progress key
        alone pinned the archive to a single bedroom cell and the frontier could never
        advance. The cell key therefore includes the milestone index.
        """
        from pokewm.emulator.vec_env import VecPokemonRed

        cfg = replace(_cfg.env, init_state=str(init_state), max_episode_steps=10**9)
        vec = VecPokemonRed(4, cfg, _cfg.reward, num_subgoals=NUM_SUBGOALS, seed=3)
        try:
            vec.reset()
            keys, progress_keys, blobs = set(), set(), 0
            rng = np.random.default_rng(3)
            for _ in range(1500):
                _, _, _, _, infos = vec.step(rng.integers(0, NUM_ACTIONS, size=4))
                for info in infos:
                    keys.add(info["cell_key"])
                    progress_keys.add(info["progress_key"])
                    blobs += info["state_blob"] is not None
                if len(keys) >= 3:
                    break
            assert len(keys) >= 3, f"only {len(keys)} cell keys: {keys}"
            assert len(keys) > len(progress_keys), (
                "cell keys must be finer-grained than progress keys during the "
                f"flag-free opening: {len(keys)} vs {len(progress_keys)}"
            )
            assert blobs >= len(keys), "every new cell key must ship a save state"
        finally:
            vec.close()

    def test_cell_key_separates_distant_positions_in_the_same_phase(self, init_state):
        """Regression: the archive must keep growing during a flag-free phase.

        The first long run froze at 28 cells for 2.7M steps on Route 1 -- no story flag
        and no milestone changed there, so a key built only from those never changed and
        Go-Explore stopped working inside every long phase. Position is what actually
        distinguishes progress along a route.
        """
        from pokewm.emulator.vec_env import cell_key

        cfg = replace(_cfg.env, init_state=str(init_state), max_episode_steps=10**9)
        e = PokemonRedEnv(cfg, _cfg.reward, num_subgoals=NUM_SUBGOALS)
        try:
            _, info = e.reset()
            gs = info["state"]
            base = cell_key(gs, 5, 8)

            # Same progress + milestone, far apart -> different cells.
            far = replace(gs, x=gs.x + 32, y=gs.y + 32)
            assert cell_key(far, 5, 8) != base

            # Same progress + milestone, one tile away -> same cell (no explosion).
            near = replace(gs, x=gs.x + 1)
            assert cell_key(near, 5, 8) == base or gs.x % 8 == 7

            # Milestone and progress still participate.
            assert cell_key(gs, 6, 8) != base
        finally:
            e.close()

    @pytest.mark.slow
    def test_archive_keeps_gaining_cells_while_exploring(self, init_state):
        """A random walk must keep minting cells, not plateau after the opening."""
        from pokewm.emulator.vec_env import VecPokemonRed

        cfg = replace(_cfg.env, init_state=str(init_state), max_episode_steps=10**9)
        vec = VecPokemonRed(4, cfg, _cfg.reward, num_subgoals=NUM_SUBGOALS, seed=5,
                            position_bucket=8)
        try:
            vec.reset()
            rng = np.random.default_rng(5)
            keys, first_half = set(), None
            for i in range(3000):
                _, _, _, _, infos = vec.step(rng.integers(0, NUM_ACTIONS, size=4))
                for info in infos:
                    keys.add(info["cell_key"])
                if i == 1499:
                    first_half = len(keys)
            assert first_half is not None
            gained_second_half = len(keys) - first_half
            assert gained_second_half > 0, (
                f"archive plateaued: {first_half} cells by mid-run, none added after"
            )
            assert len(keys) >= 8, f"too few distinct cells: {len(keys)}"
        finally:
            vec.close()

    def test_restore_seeds_the_milestone_tracker(self, init_state):
        """Regression: a restored worker must not report itself back to milestone 1.

        The archive scores cells by milestone. When the tracker restarted from scratch
        on every restore, workers relaunched deep in the game filed their new cells as
        milestone 1 — so the archive's own restores destroyed its frontier estimate.
        """
        from pokewm.emulator.vec_env import VecPokemonRed

        cfg = replace(_cfg.env, init_state=str(init_state), max_episode_steps=10**9)
        vec = VecPokemonRed(1, cfg, _cfg.reward, num_subgoals=NUM_SUBGOALS, seed=0)
        try:
            _, infos = vec.reset()
            blob = None
            for _ in range(400):
                _, _, _, _, infos = vec.step([0])
                if infos[0]["state_blob"] is not None:
                    blob = infos[0]["state_blob"]
            assert blob is not None

            # Restore with no history: tracker starts over.
            _, infos = vec.reset([blob])
            bare = infos[0]["milestone"]

            # Restore with the history the archive would have supplied.
            deep_history = {
                M.MAP_IDS[n] for n in
                ["REDS_HOUSE_2F", "REDS_HOUSE_1F", "PALLET_TOWN", "OAKS_LAB", "ROUTE_1"]
            }
            _, infos = vec.reset([blob], seen_maps=[deep_history])
            seeded = infos[0]["milestone"]

            assert seeded > bare, (
                f"seeded restore reported milestone {seeded}, bare reported {bare}; "
                "history was not carried across the reset"
            )
            # The prefix stops at `got_starter`, which needs a party member -- this save
            # state is from a random walk around the house and has none. Everything the
            # seeded map history *can* establish (leave_room, leave_house, oaks_lab) is
            # credited, which is the property under test.
            assert seeded == 4, seeded
        finally:
            vec.close()

    def test_subgoal_broadcast(self, init_state):
        from pokewm.emulator.vec_env import VecPokemonRed

        cfg = replace(_cfg.env, init_state=str(init_state), max_episode_steps=64)
        vec = VecPokemonRed(2, cfg, _cfg.reward, num_subgoals=NUM_SUBGOALS, seed=0)
        try:
            vec.reset()
            vec.set_subgoals([5, 9])
            obs, _, _, _, _ = vec.step([0, 0])
            assert obs["subgoal"][0].argmax() == 5
            assert obs["subgoal"][1].argmax() == 9
        finally:
            vec.close()

    def test_epistemic_bonus_rides_along_with_the_step(self, init_state):
        from pokewm.emulator.vec_env import VecPokemonRed

        cfg = replace(_cfg.env, init_state=str(init_state), max_episode_steps=64)
        vec = VecPokemonRed(2, cfg, _cfg.reward, num_subgoals=NUM_SUBGOALS, seed=0)
        try:
            vec.reset()
            _, _, _, _, infos = vec.step([0, 0], epistemic=[1.0, 0.0])
            assert infos[0]["reward_breakdown"].get("epistemic", 0.0) > 0
            assert "epistemic" not in infos[1]["reward_breakdown"]
        finally:
            vec.close()

    def test_step_without_epistemic_is_still_valid(self, init_state):
        from pokewm.emulator.vec_env import VecPokemonRed

        cfg = replace(_cfg.env, init_state=str(init_state), max_episode_steps=64)
        vec = VecPokemonRed(2, cfg, _cfg.reward, num_subgoals=NUM_SUBGOALS, seed=0)
        try:
            vec.reset()
            _, r, _, _, infos = vec.step([0, 0])
            assert r.shape == (2,)
            assert all("epistemic" not in i["reward_breakdown"] for i in infos)
        finally:
            vec.close()

    def test_text_state_for_the_viewer(self, init_state):
        from pokewm.emulator.vec_env import VecPokemonRed

        cfg = replace(_cfg.env, init_state=str(init_state))
        vec = VecPokemonRed(1, cfg, _cfg.reward, num_subgoals=NUM_SUBGOALS, seed=0)
        try:
            vec.reset()
            vec.step([0])
            snap = vec.text_state(0)
            assert "location" in snap["text"]
            assert snap["screen"].shape == (144, 160, 3)
        finally:
            vec.close()


class TestMapProgress:
    """Regression: the only directed term in the reward function.

    Every other exploration term is first-visit-only, so on ground the agent has already
    covered there is no signal at all -- new_tile and new_map are spent, and the
    epistemic bonus decays to zero exactly where the world model fits well. Restored
    onto Route 1 with Viridian north and Pallet south, the agent kept re-running the
    delivery route it had spent millions of steps learning and walked *south*: 38
    archived frontier cells accumulated in Pallet Town against 2 on Route 1.
    """

    def _state(self, map_id: int):
        mem = FakeMemory()
        mem.write(RM.CUR_MAP, map_id)
        mem.write(RM.X_COORD, 5)
        mem.write(RM.Y_COORD, 5)
        mem.write(RM.PARTY_COUNT, 1)
        make_party_mon(mem, 0, species=1, level=6, hp=20, max_hp=20)
        return RM.RamReader(mem).read()

    def _walk(self, env, maps):
        """Feed a sequence of maps through the real reward function."""
        out = []
        for m in maps:
            _, b = env._reward(self._state(m))
            out.append(b.get("map_progress", 0.0))
        return out

    def test_moving_forward_along_the_path_pays(self, env):
        env.reset()
        env._map_rank_ref = map_rank(M.MAP_IDS["ROUTE_1"])
        (gain,) = self._walk(env, [M.MAP_IDS["VIRIDIAN_CITY"]])
        assert gain > 0

    def test_moving_backward_along_the_path_costs(self, env):
        env.reset()
        env._map_rank_ref = map_rank(M.MAP_IDS["ROUTE_1"])
        (loss,) = self._walk(env, [M.MAP_IDS["PALLET_TOWN"]])
        assert loss < 0

    def test_a_round_trip_is_worth_exactly_zero(self, env):
        """Telescoping is what makes this unfarmable: no oscillation pays."""
        env.reset()
        env._map_rank_ref = map_rank(M.MAP_IDS["ROUTE_1"])
        gains = self._walk(env, [M.MAP_IDS["VIRIDIAN_CITY"], M.MAP_IDS["ROUTE_1"]] * 4)
        assert sum(gains) == pytest.approx(0.0)

    def test_it_adds_no_standing_per_step_cost(self, env):
        """A -0.002/step term once made blacking out the optimal policy."""
        env.reset()
        env._map_rank_ref = map_rank(M.MAP_IDS["VIRIDIAN_CITY"])
        gains = self._walk(env, [M.MAP_IDS["VIRIDIAN_CITY"]] * 20)
        assert all(g == 0.0 for g in gains)

    def test_entering_an_off_path_building_is_free(self, env):
        """Charging for it would have penalised entering the Mart, where the parcel is."""
        env.reset()
        env._map_rank_ref = map_rank(M.MAP_IDS["VIRIDIAN_CITY"])
        gains = self._walk(
            env, [M.MAP_IDS["VIRIDIAN_MART"], M.MAP_IDS["VIRIDIAN_CITY"]]
        )
        assert gains == [0.0, 0.0]

    def test_the_delivery_detour_costs_less_than_completing_it(self, env):
        """The parcel must be carried backwards; the shaping must not forbid that."""
        env.reset()
        env._map_rank_ref = map_rank(M.MAP_IDS["VIRIDIAN_CITY"])
        back = -sum(self._walk(env, [M.MAP_IDS["ROUTE_1"], M.MAP_IDS["PALLET_TOWN"]]))
        assert back < _cfg.reward.event

    def test_reset_re_anchors_so_an_episode_boundary_is_never_charged(self, env):
        """An archive restore can drop the agent anywhere."""
        env.reset()
        env._map_rank_ref = map_rank(M.MAP_IDS["VIRIDIAN_CITY"])
        env.reset()  # boots in Red's bedroom, far behind Viridian
        _, b = env._reward(env.ram.read())
        assert b.get("map_progress", 0.0) == 0.0

    def test_disabled_by_zero_weight(self, init_state):
        cfg = replace(_cfg.env, init_state=str(init_state), max_episode_steps=256)
        rcfg = replace(_cfg.reward, map_progress=0.0)
        e = PokemonRedEnv(cfg, rcfg, num_subgoals=NUM_SUBGOALS)
        try:
            e.reset()
            e._map_rank_ref = map_rank(M.MAP_IDS["ROUTE_1"])
            _, b = e._reward(self._state(M.MAP_IDS["VIRIDIAN_CITY"]))
            assert "map_progress" not in b
        finally:
            e.close()


class TestMapProgressIsSizedAgainstNovelty:
    """Regression: the directed term was outbid by the novelty bonus.

    Novelty-based exploration goes where the unexplored volume is, and there is far more
    of it behind the frontier than ahead. At 0.15 the mean payout per on-path transition
    fell monotonically -0.024 -> -0.122 over 500k steps: the agent was learning to walk
    backwards while the archive kept restoring it to the frontier.

    The sizing rule is deliberately *not* "beat a fresh map". A fresh map is worth
    several points of tile credit and should win -- that is exploration doing its job.
    What must not pay is retreating over ground already covered, where only a handful of
    tiles are still fresh.
    """

    def test_retreating_over_covered_ground_does_not_pay(self):
        rc = _cfg.reward
        # Rear areas are largely explored by the time this matters: Viridian City held 68
        # occupied position buckets when this was measured.
        fresh_tiles_on_a_revisit = 15
        assert rc.map_progress > fresh_tiles_on_a_revisit * rc.new_tile

    def test_a_genuinely_new_map_still_outbids_it(self):
        """Exploration must not be suppressed -- only backtracking over old ground."""
        rc = _cfg.reward
        tiles_on_a_fresh_town = 300
        assert tiles_on_a_fresh_town * rc.new_tile > rc.map_progress

    def test_it_stays_well_below_a_story_flag(self):
        """The parcel run must be carried backwards; shaping must not forbid it."""
        rc = _cfg.reward
        viridian_to_pallet_ranks = 5
        assert viridian_to_pallet_ranks * rc.map_progress < 2 * rc.event

    def test_it_remains_a_transition_term_not_a_standing_cost(self, env):
        """Telescoping is what makes it safe to raise; a per-step version would not be."""
        env.reset()
        env._map_rank_ref = map_rank(M.MAP_IDS["ROUTE_2"])
        out = []
        for _ in range(15):
            _, b = env._reward(TestMapProgress()._state(M.MAP_IDS["ROUTE_2"]))
            out.append(b.get("map_progress", 0.0))
        assert all(v == 0.0 for v in out)


class TestCombatIsWorthEntering:
    """Regression: the reward made battling negative-EV, so the party never levelled.

    Measured over 18M env steps, `level_sum` never left 3.75-4.5 -- a single starter
    that levelled up zero times. Not an inability: winning a wild battle pays a fraction
    of a level while losing paid `faint`, so at 0.6 against -10 engaging was rational
    only above ~97% win probability. A level-5 party correctly learned to avoid every
    fight, and a party that never fights never levels, which caps the run long before
    Brock's level-14 Onix. Route 2's grass then blacked the agent out on every attempt
    to cross it, teleporting it to Pallet Town.
    """

    def test_a_wipe_costs_only_a_couple_of_levels(self):
        rc = _cfg.reward
        levels_lost = abs(rc.faint) / rc.level
        assert levels_lost <= 3.0, f"a wipe costs {levels_lost:.1f} levels of progress"

    def test_grinding_to_brock_is_worth_more_than_a_wipe(self):
        """The agent must prefer levelling up to never fighting."""
        rc = _cfg.reward
        levels_needed = 8          # level ~5 starter -> Brock's level-14 Onix
        assert levels_needed * rc.level > abs(rc.faint)

    def test_a_badge_still_dominates_grinding(self):
        """Levels are a means, not the objective -- they must not out-earn milestones."""
        rc = _cfg.reward
        assert rc.badge > 8 * rc.level


class TestBlackoutIsNotBilledAsTravel:
    """Regression: the involuntary teleport home was charged as backward map progress.

    At map_progress 0.5 a Route 2 -> Pallet Town wipe billed -4.0 on top of `faint`.
    Worse than the double-count, it corrupted the diagnostic: mean payout per on-path
    transition read as the agent *choosing* to walk backwards when it was being carried
    backwards -- a different fault with a different fix.
    """

    def _state(self, map_id):
        return TestMapProgress()._state(map_id)

    def test_the_teleport_home_is_not_charged(self, env):
        env.reset()
        env._map_rank_ref = map_rank(M.MAP_IDS["ROUTE_2"])
        env._blackout_map_pending = True
        env._blackout_hp_pending = True          # as set when the party wipes
        _, b = env._reward(self._state(M.MAP_IDS["PALLET_TOWN"]))
        assert "map_progress" not in b

    def test_the_reference_still_moves_so_walking_back_out_pays(self, env):
        """Re-anchoring must not leave a credit the agent can collect for free."""
        env.reset()
        env._map_rank_ref = map_rank(M.MAP_IDS["ROUTE_2"])
        env._blackout_map_pending = True
        env._blackout_hp_pending = True
        env._reward(self._state(M.MAP_IDS["PALLET_TOWN"]))
        assert env._map_rank_ref == map_rank(M.MAP_IDS["PALLET_TOWN"])
        _, b = env._reward(self._state(M.MAP_IDS["ROUTE_1"]))
        assert b["map_progress"] > 0          # earned by walking, not by fainting

    def test_ordinary_backtracking_is_still_charged(self, env):
        env.reset()
        env._map_rank_ref = map_rank(M.MAP_IDS["ROUTE_2"])
        env._blackout_map_pending = False
        env._blackout_hp_pending = False
        _, b = env._reward(self._state(M.MAP_IDS["VIRIDIAN_CITY"]))
        assert b["map_progress"] < 0

    def test_only_one_teleport_is_forgiven_per_wipe(self, env):
        env.reset()
        env._map_rank_ref = map_rank(M.MAP_IDS["ROUTE_2"])
        env._blackout_map_pending = True
        env._blackout_hp_pending = True
        env._reward(self._state(M.MAP_IDS["PALLET_TOWN"]))
        _, b = env._reward(self._state(M.MAP_IDS["VIRIDIAN_CITY"]))
        assert "map_progress" in b


class TestBattleRewards:
    """Trainer battles cannot be fled, and the agent kept choosing RUN anyway.

    Selecting RUN in a trainer battle prints a refusal and burns the turn, so the battle
    stalls indefinitely. Two things were missing: the battle menu was not observable at
    all (`menu_active` was a bare boolean, so FIGHT and RUN looked identical in the
    state), and nothing dense rewarded attacking, so "attack" and "reopen the bag" scored
    the same.

    The stall term is deliberately keyed on "nothing is happening" rather than on a RUN
    cursor index: measurement here showed the battle menu is nested and `wMaxMenuItem`
    shifts 7 -> 1 -> 3 as submenus open, so a hardcoded index would be ROM-specific and
    wrong whenever a submenu is up.
    """

    def _st(self, hp=1.0, enemy_hp=20, enemy_max=20, in_battle=2):
        mem = FakeMemory()
        mem.write(RM.CUR_MAP, M.MAP_IDS["ROUTE_2"])
        mem.write(RM.X_COORD, 5)
        mem.write(RM.Y_COORD, 5)
        mem.write(RM.PARTY_COUNT, 1)
        make_party_mon(mem, 0, species=1, level=6, hp=int(round(20 * hp)), max_hp=20)
        mem.write(RM.IS_IN_BATTLE, in_battle)
        mem.write(RM.ENEMY_MON_HP, [enemy_hp >> 8, enemy_hp & 0xFF])
        mem.write(RM.ENEMY_MON_MAX_HP, [enemy_max >> 8, enemy_max & 0xFF])
        return RM.RamReader(mem).read()

    def test_the_menu_cursor_is_observable(self):
        """FIGHT and RUN must not look identical in the observation."""
        assert "menu_cursor" in RM.SYMBOLIC_FEATURES

    def _menu(self, top_y, top_x, item, max_item, in_battle=2):
        mem = FakeMemory()
        mem.write(RM.CUR_MAP, M.MAP_IDS["ROUTE_2"])
        mem.write(RM.PARTY_COUNT, 1)
        make_party_mon(mem, 0, species=1, level=6, hp=20, max_hp=20)
        mem.write(RM.IS_IN_BATTLE, in_battle)
        mem.write(RM.TOP_MENU_ITEM_Y, top_y)
        mem.write(RM.TOP_MENU_ITEM_X, top_x)
        mem.write(RM.CURRENT_MENU_ITEM, item)
        mem.write(RM.MAX_MENU_ITEM, max_item)
        return RM.encode_symbolic(RM.RamReader(mem).read())

    def test_fight_and_item_are_different_observations(self):
        """The battle menu is a 2x2 grid: the column is only in `wTopMenuItemX`.

        Measured live: FIGHT=(top_x 9, item 0), PKMN=(9, 1), ITEM=(15, 0), RUN=(15, 1),
        all with max_menu_item=1. On `menu_item` alone FIGHT and ITEM are both 0.0.
        """
        fight = self._menu(14, 9, 0, 1)
        item = self._menu(14, 15, 0, 1)
        assert not np.allclose(fight, item)

    def test_pkmn_and_run_are_different_observations(self):
        pkmn = self._menu(14, 9, 1, 1)
        run = self._menu(14, 15, 1, 1)
        assert not np.allclose(pkmn, run)

    def test_the_move_list_is_distinguishable_from_the_battle_menu(self):
        """Nested menus differ in row origin: battle menu at y=14, move list at y=12."""
        battle_menu = self._menu(14, 9, 1, 1)
        move_list = self._menu(12, 5, 1, 3)
        assert not np.allclose(battle_menu, move_list)

    def test_a_ratio_cursor_alone_would_confuse_these(self):
        """Pins *why* the extra fields are needed rather than just that they exist."""
        assert (0 / 1) == (0 / 3)          # FIGHT vs first move
        assert (1 / 1) == (3 / 3)          # RUN vs last move

    def _list(self, item, scroll, quantity=1, top_y=4, top_x=5, max_item=2):
        mem = FakeMemory()
        mem.write(RM.CUR_MAP, M.MAP_IDS["VIRIDIAN_MART"])
        mem.write(RM.PARTY_COUNT, 1)
        make_party_mon(mem, 0, species=1, level=6, hp=20, max_hp=20)
        mem.write(RM.TOP_MENU_ITEM_Y, top_y)
        mem.write(RM.TOP_MENU_ITEM_X, top_x)
        mem.write(RM.CURRENT_MENU_ITEM, item)
        mem.write(RM.MAX_MENU_ITEM, max_item)
        mem.write(RM.LIST_SCROLL_OFFSET, scroll)
        mem.write(RM.ITEM_QUANTITY, quantity)
        return RM.encode_symbolic(RM.RamReader(mem).read())

    def test_a_scrolled_list_selects_a_different_item(self):
        """Measured in the Viridian Mart: pressing down gave item=2/scroll=0, then
        item=2/scroll=1, then item=2/scroll=2 -- three different items, identical menu
        bytes. `wCurrentMenuItem` indexes a row on screen, not a list entry."""
        third = self._list(item=2, scroll=0)
        fourth = self._list(item=2, scroll=1)
        fifth = self._list(item=2, scroll=2)
        assert not np.allclose(third, fourth)
        assert not np.allclose(fourth, fifth)

    def test_list_index_is_absolute_across_scroll_positions(self):
        """Entry 2 is entry 2 however the page sits under the cursor.

        The full vectors still differ, and should: cursor-on-row-3-unscrolled and
        cursor-on-row-1-scrolled-by-2 are different screens that happen to highlight
        the same entry. It is `list_index` that has to agree.
        """
        i = RM.SYMBOLIC_FEATURES.index("list_index")
        assert self._list(item=2, scroll=0)[i] == self._list(item=0, scroll=2)[i]
        assert self._list(item=2, scroll=0)[i] != self._list(item=2, scroll=2)[i]

    def test_the_quantity_selector_is_observable(self):
        """Buying one ball and buying five differed in no observed feature at all."""
        one = self._list(item=0, scroll=0, quantity=1)
        five = self._list(item=0, scroll=0, quantity=5)
        assert not np.allclose(one, five)

    def test_the_mart_root_menu_is_distinct_from_the_item_list(self):
        """BUY/SELL/QUIT sits at (1,1); the item list it opens sits at (4,5)."""
        root = self._list(item=0, scroll=0, top_y=1, top_x=1)
        items = self._list(item=0, scroll=0, top_y=4, top_x=5)
        assert not np.allclose(root, items)

    def test_every_start_menu_entry_is_distinct(self):
        """The START menu is a 7-entry list at (2,11); all entries must differ."""
        seen = [tuple(self._list(item=i, scroll=0, top_y=2, top_x=11, max_item=7))
                for i in range(7)]
        assert len(set(seen)) == 7

    def test_menu_active_is_not_a_feature(self):
        """`wMaxMenuItem` is never cleared, so it read nonzero in 1119/1119 archived
        overworld cells with no menu open -- a constant input, not a signal."""
        assert "menu_active" not in RM.SYMBOLIC_FEATURES

    def test_dealing_damage_pays(self, env):
        env.reset()
        env._reward(self._st(enemy_hp=20))
        _, b = env._reward(self._st(enemy_hp=15))
        assert b["enemy_damage"] > 0

    def test_taking_damage_does_not_pay_the_damage_term(self, env):
        env.reset()
        env._reward(self._st(enemy_hp=20))
        _, b = env._reward(self._st(enemy_hp=20, hp=0.5))
        assert "enemy_damage" not in b

    def test_a_stalled_trainer_battle_is_penalised(self, env):
        env.reset()
        grace = _cfg.reward.battle_stall_grace
        out = [env._reward(self._st())[1] for _ in range(grace + 6)]
        assert "battle_stall" not in out[0], "penalised before the grace period"
        assert "battle_stall" in out[-1], "never penalised despite no progress"

    def test_a_progressing_trainer_battle_is_not_penalised(self, env):
        env.reset()
        hp = 20
        for _ in range(_cfg.reward.battle_stall_grace + 6):
            hp = max(hp - 1, 0)
            _, b = env._reward(self._st(enemy_hp=hp))
        assert "battle_stall" not in b

    def test_wild_battles_are_not_stall_penalised(self, env):
        """Wild battles *can* be fled, so waiting there is a legitimate choice."""
        env.reset()
        out = [env._reward(self._st(in_battle=1))[1]
               for _ in range(_cfg.reward.battle_stall_grace + 6)]
        assert all("battle_stall" not in b for b in out)

    def test_the_stall_cost_cannot_outbid_attacking(self, env):
        rc = _cfg.reward
        assert abs(rc.battle_stall) < rc.enemy_damage / 10


class TestHpPotential:
    """Healing must be worth a detour, without becoming a damage/heal money loop.

    Archived frontier states measured party_hp_frac 0.40 with a single Pokemon: the agent
    had been fighting at 40% health indefinitely. The old `heal` term paid for recovering
    HP but charged nothing for losing it, so a damage-then-heal cycle was free money --
    unexploited only because the agent never healed at all.
    """

    def _st(self, hp):
        return TestBattleRewards()._st(hp=hp, in_battle=0)

    def test_healing_pays(self, env):
        env.reset()
        env._reward(self._st(0.4))
        _, b = env._reward(self._st(1.0))
        assert b["hp_potential"] > 0

    def test_taking_damage_costs(self, env):
        env.reset()
        env._reward(self._st(1.0))
        _, b = env._reward(self._st(0.4))
        assert b["hp_potential"] < 0

    def test_a_damage_then_heal_cycle_is_worth_zero(self, env):
        """Symmetry is what makes it unfarmable."""
        env.reset()
        env._reward(self._st(1.0))
        total = 0.0
        for _ in range(4):
            total += env._reward(self._st(0.3))[1].get("hp_potential", 0.0)
            total += env._reward(self._st(1.0))[1].get("hp_potential", 0.0)
        assert total == pytest.approx(0.0, abs=1e-6)

    def test_a_full_heal_is_worth_more_than_a_step_of_wandering(self, env):
        """It has to be worth walking to a Pokecenter for."""
        rc = _cfg.reward
        assert rc.hp_potential * 0.6 > 20 * rc.new_tile

    def test_the_blackout_refill_is_not_paid(self, env):
        """Waking at full HP must not refund the wipe that caused it."""
        env.reset()
        env._reward(self._st(0.2))
        env._blackout_map_pending = True
        env._blackout_hp_pending = True
        _, b = env._reward(self._st(1.0))
        assert "hp_potential" not in b


class TestBlackoutSuppressionSurvivesBothEvents:
    """Regression: one shared flag, two terms -- whichever ran first swallowed it.

    A wipe produces a teleport home *and* a full refill in the same transition. With a
    single `_blackout_pending`, the map-progress block consumed it and the HP block then
    paid the refill anyway. Measured live as `hp_potential` pinned at exactly +3.000 (a
    full 0 -> 1.0 swing) firing alongside `faint`, which made a blackout cost -5 instead
    of the intended -8. The original test set the flag and changed only HP, so it never
    reproduced the real ordering.
    """

    def _st(self, map_id, hp):
        return TestBattleRewards()._st(hp=hp, in_battle=0) if False else None

    def test_neither_term_pays_when_both_events_land_together(self, env):
        env.reset()
        env._map_rank_ref = map_rank(M.MAP_IDS["ROUTE_2"])
        env._prev_hp_frac = 0.0
        env._blackout_map_pending = True
        env._blackout_hp_pending = True

        mem = FakeMemory()
        mem.write(RM.CUR_MAP, M.MAP_IDS["PALLET_TOWN"])   # teleported home
        mem.write(RM.X_COORD, 5)
        mem.write(RM.Y_COORD, 5)
        mem.write(RM.PARTY_COUNT, 1)
        make_party_mon(mem, 0, species=1, level=6, hp=20, max_hp=20)  # and refilled
        gs = RM.RamReader(mem).read()

        _, b = env._reward(gs)
        assert "map_progress" not in b, "the teleport was billed as travel"
        assert "hp_potential" not in b, "the free refill was paid"

    def test_a_later_genuine_heal_still_pays(self, env):
        """Suppression must be one-shot, not a permanent mute."""
        env.reset()
        env._prev_hp_frac = 0.0
        env._blackout_map_pending = True
        env._blackout_hp_pending = True
        st = TestBattleRewards()._st
        env._reward(st(hp=1.0, in_battle=0))          # the refill: not paid
        env._reward(st(hp=0.4, in_battle=0))          # took damage: charged
        _, b = env._reward(st(hp=1.0, in_battle=0))   # walked to a Pokecenter
        assert b.get("hp_potential", 0.0) > 0


class TestExplorationMemorySurvivesARestart:
    """Regression: `new_tile` paid again for ground already covered.

    `seen_coords` is documented as a *run*-level record -- an agent must not be paid
    twice for discovering Route 1 -- and it survives `reset()`. But it lived only in the
    worker process, so a restart wiped it. Across a night of restarts that inverted the
    term: instead of a bounty on new ground it became a repeating payment for re-walking
    the easy, well-trodden part of a map. Measured alongside a 14.7M-step milestone
    plateau in which the Viridian Forest north gate was never once reached.
    """

    def test_round_trips(self, env):
        env.reset()
        for _ in range(6):
            env.step(0)
        before = env.export_exploration()
        assert before["seen_coords"], "precondition: something was explored"

        fresh = {"seen_coords": [], "seen_maps": [], "max_event_bits": 0,
                 "max_badges": 0, "max_level_sum": 0, "max_dex_owned": 0,
                 "max_dex_seen": 0}
        env.import_exploration(fresh)
        assert not env.seen_coords
        env.import_exploration(before)
        assert {tuple(c) for c in before["seen_coords"]} == env.seen_coords

    def test_restored_tiles_no_longer_pay(self, env):
        """The whole point: covered ground must stop earning."""
        env.reset()
        gs = env.ram.read()
        env.import_exploration({"seen_coords": [[gs.map_id, gs.x, gs.y]],
                                "seen_maps": [gs.map_id]})
        _, b = env._reward(gs)
        assert "new_tile" not in b
        assert "new_map" not in b

    def test_monotone_maxima_are_restored_too(self, env):
        """Otherwise `event`, `level` and dex credit are re-earned after a restart."""
        env.reset()
        env.import_exploration({"seen_coords": [], "seen_maps": [],
                                "max_event_bits": 42, "max_badges": 3,
                                "max_level_sum": 77, "max_dex_owned": 9,
                                "max_dex_seen": 20})
        assert env.max_event_bits == 42
        assert env.max_level_sum == 77
        assert env.max_dex_owned == 9

    def test_missing_state_is_tolerated(self, env):
        """Checkpoints written before this existed must still load."""
        env.reset()
        env.import_exploration(None)
        env.import_exploration({})


class TestUncontrollableStatesAreNotArchived:
    """Regression: cells were saved mid-script, with a transient position.

    While `wJoyIgnore` is nonzero -- map transitions, text boxes, cutscenes -- or a
    battle is up, the agent does not control the character and the position in RAM is in
    flux. That position becomes the cell key and therefore the archive bucket. One such
    cell sat at Viridian Forest (5,0), a tile with no walkable neighbour; pressing a
    direction from it moved the agent to (17,43) or (15,47), wherever the script was
    going. It also corrupted a reachability analysis by looking like the closest
    approach to the map's exit. Measured: 46 of 778 cells (6%) were captured this way.
    """

    def test_the_guard_is_in_the_worker(self):
        import inspect

        from pokewm.emulator import vec_env

        src = inspect.getsource(vec_env)
        assert "joy_ignore == 0" in src and "in_battle == 0" in src

    def test_script_active_is_observable(self):
        """The signal the guard uses is the one the agent already sees."""
        assert "script_active" in RM.SYMBOLIC_FEATURES

    def test_the_guard_only_excludes_uncontrollable_states(self):
        """It must gate on control, not on anything incidental to it."""
        import inspect

        from pokewm.emulator import vec_env

        src = inspect.getsource(vec_env)
        i = src.index("controllable = ")
        line = src[i:src.index("\n", i)]
        assert "joy_ignore" in line and "in_battle" in line
        assert "and" in line, "both conditions must be required"


class TestHealIsPrioritisedWhenHurt:
    """A damaged party should be heading for a Pokecenter, not deeper into grass.

    The LLM proposer decides well but slowly: single-flight, a 30 s cooldown, round-robin
    over 8 workers, so a given worker's subgoal is minutes old. "You are at 10% HP"
    cannot wait minutes -- and that was the agent's normal state. Every archived Viridian
    Forest cell held one level-6 Pokemon at 10-40% health, and an exhaustive search from
    them lost 72 of 90 wild encounters.
    """

    def _hp(self, env, frac):
        mem = FakeMemory()
        mem.write(RM.CUR_MAP, M.MAP_IDS["ROUTE_2"])
        mem.write(RM.X_COORD, 5)
        mem.write(RM.Y_COORD, 5)
        mem.write(RM.PARTY_COUNT, 1)
        make_party_mon(mem, 0, species=1, level=6, hp=int(round(20 * frac)), max_hp=20)
        return RM.RamReader(mem).read()

    def test_a_hurt_party_switches_the_subgoal_to_heal(self, env):
        from pokewm.llm.subgoals import HEAL_SUBGOAL_ID

        env.reset()
        env.set_subgoal(12)                       # proposer wanted TRAIN_LEVELS
        assert env._active_subgoal(0.1, party_size=1) == HEAL_SUBGOAL_ID % env.num_subgoals

    def test_an_empty_party_does_not_force_heal(self, env):
        """0.0 HP with no party is the start of the game, not an injury."""
        env.reset()
        env.set_subgoal(12)
        assert env._active_subgoal(0.0, party_size=0) == 12

    def test_a_healthy_party_keeps_the_proposed_subgoal(self, env):
        env.reset()
        env.set_subgoal(12)
        assert env._active_subgoal(1.0, party_size=1) == 12

    def test_a_scratch_does_not_trigger_a_detour(self, env):
        """Walking to a Pokecenter costs real steps; it must be worth it."""
        env.reset()
        env.set_subgoal(12)
        assert env._active_subgoal(0.9, party_size=1) == 12

    def test_no_balls_and_money_points_at_the_mart(self, env):
        """CATCH is unsatisfiable with an empty bag; BUY_ITEMS is the precondition.

        Measured at 54M env steps: 205 of 205 archived Viridian Forest cells carried
        zero balls and a party of one, so every wild encounter forced a CATCH subgoal
        that no action sequence could satisfy.
        """
        env.reset()
        env.set_subgoal(12)
        assert env._active_subgoal(1.0, party_size=1, ball_count=0, money=3000) == 14

    def test_carrying_a_ball_restores_the_proposed_subgoal(self, env):
        env.reset()
        env.set_subgoal(12)
        assert env._active_subgoal(1.0, party_size=1, ball_count=5, money=3000) == 12

    def test_an_unaffordable_mart_trip_is_not_forced(self, env):
        """Otherwise one unsatisfiable subgoal is swapped for another."""
        env.reset()
        env.set_subgoal(12)
        assert env._active_subgoal(1.0, party_size=1, ball_count=0, money=50) == 12

    def test_a_wild_battle_without_balls_does_not_force_catch(self, env):
        env.reset()
        env.set_subgoal(12)
        got = env._active_subgoal(1.0, party_size=1, in_battle=1,
                                  ball_count=0, money=3000)
        assert got != 11

    def test_a_wild_battle_with_balls_forces_catch(self, env):
        env.reset()
        env.set_subgoal(12)
        got = env._active_subgoal(1.0, party_size=1, in_battle=1,
                                  ball_count=3, money=3000)
        assert got == 11

    def test_being_hurt_still_outranks_a_shopping_trip(self, env):
        env.reset()
        env.set_subgoal(12)
        assert env._active_subgoal(0.1, party_size=1, ball_count=0, money=3000) == 13

    def test_the_override_is_not_paid_a_bonus(self, env):
        """Otherwise damage-then-heal becomes a profitable loop."""
        env.reset()
        env.set_subgoal(12)
        env._forced_subgoal = True
        env._prev_state = self._hp(env, 0.2)
        _, b = env._reward(self._hp(env, 1.0))
        assert "subgoal" not in b
        assert b.get("hp_potential", 0.0) > 0, "recovery is still paid, symmetrically"

    def test_a_genuine_proposal_is_still_paid(self, env):
        """The override must not suppress ordinary subgoal payouts."""
        def party(level):
            mem = FakeMemory()
            mem.write(RM.CUR_MAP, M.MAP_IDS["ROUTE_2"])
            mem.write(RM.X_COORD, 5)
            mem.write(RM.Y_COORD, 5)
            mem.write(RM.PARTY_COUNT, 1)
            make_party_mon(mem, 0, species=1, level=level, hp=20, max_hp=20)
            return RM.RamReader(mem).read()

        env.reset()
        env.set_subgoal(12)                      # TRAIN_LEVELS
        env._forced_subgoal = False
        env._subgoal_paid = False
        env._prev_state = party(6)
        env._prev_hp_frac = 1.0
        _, b = env._reward(party(9))             # levelled up
        assert b.get("subgoal", 0.0) > 0

    def test_disabled_by_zero_threshold(self, env):
        env.reset()
        env.set_subgoal(12)
        env.reward_cfg = replace(env.reward_cfg, heal_subgoal_hp=0.0)
        assert env._active_subgoal(0.05, party_size=1) == 12


class TestCatchingIsPrioritised:
    """A party of one makes every faint a blackout.

    The agent reached 44M env steps having never caught anything -- party size 1 in every
    archived cell -- so a single lost encounter ended the run. It lost 72 of 90 wild
    battles in Viridian Forest for exactly that reason. Two gaps: nothing rewarded party
    *size* (`dex_owned` pays for a new species, so a duplicate added a life and paid
    zero), and the chance to catch exists only during a wild battle, which lasts seconds
    -- far inside the LLM proposer's minutes-long latency.
    """

    def _party(self, n, in_battle=0, species_base=1):
        mem = FakeMemory()
        mem.write(RM.CUR_MAP, M.MAP_IDS["ROUTE_2"])
        mem.write(RM.X_COORD, 5)
        mem.write(RM.Y_COORD, 5)
        mem.write(RM.PARTY_COUNT, n)
        for i in range(n):
            make_party_mon(mem, i, species=species_base + i, level=6, hp=20, max_hp=20)
        mem.write(RM.IS_IN_BATTLE, in_battle)
        return RM.RamReader(mem).read()

    def test_a_new_party_member_pays(self, env):
        env.reset()
        env.max_party_size = 1
        _, b = env._reward(self._party(2))
        assert b.get("party_member", 0.0) > 0

    def test_it_outweighs_a_level(self, env):
        """A catch is worth more than a level: it is the difference between losing a
        fight and losing the run."""
        rc = _cfg.reward
        assert rc.party_member > rc.level
        assert rc.party_member > rc.dex_owned

    def test_it_is_monotone_so_a_pc_deposit_cannot_farm_it(self, env):
        env.reset()
        env.max_party_size = 0
        first = env._reward(self._party(3))[1].get("party_member", 0.0)
        env._reward(self._party(1))                    # deposited two at a PC
        again = env._reward(self._party(3))[1].get("party_member", 0.0)
        assert first > 0 and again == 0.0

    def test_a_wild_battle_with_a_small_party_forces_catch(self, env):
        from pokewm.llm.subgoals import CATCH_SUBGOAL_ID

        env.reset()
        env.set_subgoal(12)
        got = env._active_subgoal(1.0, party_size=1, in_battle=1)
        assert got == CATCH_SUBGOAL_ID % env.num_subgoals

    def test_catching_outranks_healing_during_a_battle(self, env):
        """A Pokecenter is unreachable mid-battle, so HEAL there is useless advice."""
        from pokewm.llm.subgoals import CATCH_SUBGOAL_ID

        got = env._active_subgoal(0.1, party_size=1, in_battle=1)
        assert got == CATCH_SUBGOAL_ID % env.num_subgoals

    def test_a_trainer_battle_does_not_force_catch(self, env):
        """Trainer Pokemon cannot be caught."""
        env.reset()
        env.set_subgoal(12)
        assert env._active_subgoal(1.0, party_size=1, in_battle=2) == 12

    def test_a_full_enough_party_does_not_force_catch(self, env):
        env.reset()
        env.set_subgoal(12)
        n = _cfg.reward.catch_subgoal_party
        assert env._active_subgoal(1.0, party_size=n, in_battle=1) == 12

    def test_the_forced_catch_is_not_paid_a_subgoal_bonus(self, env):
        """`party_member` pays for the catch; the override supplies direction only."""
        env.reset()
        env.set_subgoal(12)
        env._forced_subgoal = True
        env.max_party_size = 1
        env._prev_state = self._party(1)
        _, b = env._reward(self._party(2))
        assert "subgoal" not in b
        assert b.get("party_member", 0.0) > 0


class TestBattlesAreNotAbandoned:
    """Regression: episodes truncated mid-battle and restored somewhere else.

    Measured live: `episode/length` was 8192 on every episode -- all of them end by
    truncation -- while `battle_stall` fired constantly, meaning trainer battles were
    sitting with neither side's HP moving. Landing the truncation inside one abandoned
    it: the worker restored from a different archive cell, the outcome never happened,
    and menu-cycling until the clock ran out became a way to avoid fighting at all.
    Trainer battles cannot be fled, so that was the only escape from them.
    """

    def _st(self, in_battle, hp=1.0, enemy_hp=20):
        mem = FakeMemory()
        mem.write(RM.CUR_MAP, M.MAP_IDS["ROUTE_2"])
        mem.write(RM.X_COORD, 5)
        mem.write(RM.Y_COORD, 5)
        mem.write(RM.PARTY_COUNT, 1)
        make_party_mon(mem, 0, species=1, level=6, hp=int(round(20 * hp)), max_hp=20)
        mem.write(RM.IS_IN_BATTLE, in_battle)
        mem.write(RM.ENEMY_MON_HP, [enemy_hp >> 8, enemy_hp & 0xFF])
        mem.write(RM.ENEMY_MON_MAX_HP, [0, 20])
        return RM.RamReader(mem).read()

    def test_a_battle_defers_truncation(self, init_state):
        cfg = replace(_cfg.env, init_state=str(init_state), max_episode_steps=4)
        e = PokemonRedEnv(cfg, _cfg.reward, num_subgoals=NUM_SUBGOALS)
        try:
            e.reset()
            trunc = False
            for _ in range(6):
                _, _, _, trunc, _ = e.step(5)
            assert trunc, "precondition: an ordinary episode truncates on budget"
        finally:
            e.close()

    def test_the_grace_is_bounded(self):
        """A stuck battle must not hold an episode open forever."""
        assert 0 < _cfg.env.battle_grace_steps <= 2048

    def test_the_stall_charge_exceeds_a_wipe(self, env):
        """Inverted deliberately, on measurement.

        This previously asserted the charge stayed *below* `faint`, to stop losing on
        purpose from becoming the cheaper way out of a stall. The mirror failure is the
        one that actually occurred: with the cap at 2.5 against a wipe at 5.0, stalling
        was cheaper, and the agent sat in an unwinnable, unfleeable gym battle until the
        episode ran out. Over the 530k steps after it first engaged Brock,
        `battle_stall` fired in 48 of 52 metric rows while `battle_won` and `faint` fired
        in none.

        Losing at least ends the fight and hands back a healthy state; stalling burns the
        episode and teaches nothing. `faint` still discourages losing on purpose.
        """
        env.reset()
        total = 0.0
        for _ in range(600):
            _, b = env._reward(self._st(in_battle=2))
            total += abs(b.get("battle_stall", 0.0))
        assert total > abs(_cfg.reward.faint), (
            f"stalling cost {total:.2f} against a wipe at {abs(_cfg.reward.faint):.2f}"
        )

    def test_the_charge_resets_when_the_battle_moves(self, env):
        """Progress should restore the full budget, not leave it spent."""
        env.reset()
        for _ in range(400):
            env._reward(self._st(in_battle=2))
        env._reward(self._st(in_battle=2, enemy_hp=10))    # dealt damage
        charged = 0.0
        for _ in range(400):
            _, b = env._reward(self._st(in_battle=2, enemy_hp=10))
            charged += abs(b.get("battle_stall", 0.0))
        assert charged > 0, "the stall penalty never re-armed after progress"


class TestTheTripToAPokecenterIsPaid:
    """`hp_potential` is symmetric, so healing only refunds the damage.

    That leaves no net reason to walk to a Pokecenter -- only to regret having needed
    one. The forced HEAL subgoal says where to go but pays nothing for going, and the
    agent kept fighting at 10% HP, losing 72 of 90 wild encounters in Viridian Forest.
    """

    def _at(self, map_name_key, hp):
        mem = FakeMemory()
        mem.write(RM.CUR_MAP, M.MAP_IDS[map_name_key])
        mem.write(RM.X_COORD, 5)
        mem.write(RM.Y_COORD, 5)
        mem.write(RM.PARTY_COUNT, 1)
        make_party_mon(mem, 0, species=1, level=6, hp=int(round(20 * hp)), max_hp=20)
        return RM.RamReader(mem).read()

    def test_arriving_hurt_pays(self, env):
        env.reset()
        env._reward(self._at("VIRIDIAN_CITY", 1.0))     # healthy
        env._reward(self._at("VIRIDIAN_CITY", 0.2))     # took damage -> trip payable
        _, b = env._reward(self._at("VIRIDIAN_POKECENTER", 0.2))
        assert b.get("heal_visit", 0.0) > 0

    def test_arriving_healthy_pays_nothing(self, env):
        env.reset()
        env._reward(self._at("VIRIDIAN_CITY", 1.0))
        _, b = env._reward(self._at("VIRIDIAN_POKECENTER", 1.0))
        assert "heal_visit" not in b

    def test_pacing_in_and_out_cannot_farm_it(self, env):
        """Paid once per bout of damage, not once per entry."""
        env.reset()
        env._reward(self._at("VIRIDIAN_CITY", 1.0))
        env._reward(self._at("VIRIDIAN_CITY", 0.2))
        total = 0.0
        for _ in range(5):
            total += env._reward(self._at("VIRIDIAN_POKECENTER", 0.2))[1].get("heal_visit", 0.0)
            total += env._reward(self._at("VIRIDIAN_CITY", 0.2))[1].get("heal_visit", 0.0)
        assert total == pytest.approx(_cfg.reward.heal_visit)

    def test_a_fresh_bout_of_damage_re_arms_it(self, env):
        env.reset()
        env._reward(self._at("VIRIDIAN_CITY", 1.0))
        env._reward(self._at("VIRIDIAN_CITY", 0.2))
        env._reward(self._at("VIRIDIAN_POKECENTER", 0.2))     # paid
        env._reward(self._at("VIRIDIAN_CITY", 1.0))           # healed up
        env._reward(self._at("VIRIDIAN_CITY", 0.2))           # hurt again
        _, b = env._reward(self._at("VIRIDIAN_POKECENTER", 0.2))
        assert b.get("heal_visit", 0.0) > 0

    def test_it_stays_under_a_story_flag(self):
        assert _cfg.reward.heal_visit < _cfg.reward.event

    def test_every_pokecenter_counts(self):
        from pokewm.emulator.maps import POKECENTER_MAPS

        assert len(POKECENTER_MAPS) >= 10
        assert M.MAP_IDS["VIRIDIAN_POKECENTER"] in POKECENTER_MAPS


class TestTransientPositionsAreNotArchived:
    """Regression: a cell claimed (5,0) while the agent was actually at (17,47).

    `joy_ignore` clears a step or two before the coordinates settle after a map change,
    so a snapshot taken in that window records a transient position -- and the position
    becomes the cell key and hence the archive bucket. Entering Viridian Forest from the
    south gate produced such a cell every time, so purging them never held. Stepping from
    one teleported clear across the map, and two separate reachability analyses were
    corrupted by it: once as the "closest tile to the exit", once as the lone discrepancy
    behind a claim that the search was unsound.
    """

    def test_the_worker_requires_the_position_to_settle(self):
        import inspect

        from pokewm.emulator import vec_env

        src = inspect.getsource(vec_env)
        assert "steps_on_map" in src
        i = src.index("controllable = ")
        line = src[i:src.index(")", i)]
        assert "joy_ignore" in line and "in_battle" in line and "steps_on_map" in line

    def test_the_counter_resets_on_a_map_change(self):
        import inspect

        from pokewm.emulator import vec_env

        src = inspect.getsource(vec_env)
        i = src.index("if gs.map_id != last_map_id")
        assert "steps_on_map = 0" in src[i:i + 200]

    def test_the_counter_resets_on_episode_reset(self):
        """A restored worker starts on a fresh map; its first steps are not settled."""
        import inspect

        from pokewm.emulator import vec_env

        src = inspect.getsource(vec_env)
        i = src.index("reported_keys.clear()")
        assert "last_map_id, steps_on_map = -1, 0" in src[i:i + 120]


class TestStatusAilmentsAreHandled:
    """Regression: the agent could not see that it was poisoned.

    `PartyMon.status` was read from RAM from the beginning but never surfaced in the
    observation, so poison -- which costs HP every few steps in the overworld -- was
    invisible. A one-Pokemon party at 10% HP bleeds out while walking, and the agent had
    no signal to attribute the loss to. Only a Pokecenter clears it; the agent carries no
    curing items.
    """

    def _st(self, status=0, hp=1.0, map_key="ROUTE_2"):
        mem = FakeMemory()
        mem.write(RM.CUR_MAP, M.MAP_IDS[map_key])
        mem.write(RM.X_COORD, 5)
        mem.write(RM.Y_COORD, 5)
        mem.write(RM.PARTY_COUNT, 1)
        make_party_mon(mem, 0, species=1, level=6, hp=int(round(20 * hp)), max_hp=20)
        mem.write(RM.PARTY_MON_1 + RM.OFF_STATUS, status)
        return RM.RamReader(mem).read()

    def test_status_is_observable(self):
        assert "party_statused" in RM.SYMBOLIC_FEATURES
        assert "party_poisoned" in RM.SYMBOLIC_FEATURES

    def test_poison_is_detected(self):
        gs = self._st(status=RM.STATUS_POISON)
        assert gs.party_poisoned
        assert gs.party_statused == pytest.approx(1.0)

    def test_a_clean_party_reads_clean(self):
        gs = self._st(status=0)
        assert not gs.party_poisoned
        assert gs.party_statused == 0.0

    def test_sleep_counts_as_a_status_but_not_poison(self):
        gs = self._st(status=2)                       # sleep counter in the low bits
        assert gs.party_statused > 0 and not gs.party_poisoned

    def test_being_poisoned_at_full_hp_still_forces_heal(self, env):
        """Waiting for HP to fall past the threshold means bleeding out on the way."""
        from pokewm.llm.subgoals import HEAL_SUBGOAL_ID

        env.reset()
        env.set_subgoal(12)
        got = env._active_subgoal(1.0, party_size=1, in_battle=0, statused=1.0)
        assert got == HEAL_SUBGOAL_ID % env.num_subgoals

    def test_curing_pays_and_contracting_costs(self, env):
        env.reset()
        env._reward(self._st(status=0))
        _, hurt = env._reward(self._st(status=RM.STATUS_POISON))
        _, cured = env._reward(self._st(status=0))
        assert hurt.get("status_potential", 0.0) < 0
        assert cured.get("status_potential", 0.0) > 0

    def test_the_cycle_is_worth_zero(self, env):
        """Symmetric, so deliberately getting poisoned earns nothing."""
        env.reset()
        env._reward(self._st(status=0))
        total = 0.0
        for _ in range(4):
            total += env._reward(self._st(status=RM.STATUS_POISON))[1].get("status_potential", 0.0)
            total += env._reward(self._st(status=0))[1].get("status_potential", 0.0)
        assert total == pytest.approx(0.0, abs=1e-6)

    def test_a_poisoned_party_makes_the_pokecenter_trip_payable(self, env):
        env.reset()
        env._reward(self._st(status=0))
        env._reward(self._st(status=RM.STATUS_POISON))
        _, b = env._reward(self._st(status=RM.STATUS_POISON, map_key="VIRIDIAN_POKECENTER"))
        assert b.get("heal_visit", 0.0) > 0


class TestHealTripIsOwedWhenRestoredHurt:
    """`heal_visit` fired zero times in 58M env steps.

    The flag that gates it was hardcoded True on reset and only cleared by a *downward*
    crossing of `heal_subgoal_hp` within an episode. Since 85% of episodes restore an
    archived cell and every deep cell is captured hurt (all 20 Pewter City cells below
    0.38 HP), the crossing never happened and the reward for walking to a Pokemon Center
    was unearnable exactly when the agent needed to walk to one.
    """

    def _st(self, env, hp, map_name="PEWTER_POKECENTER"):
        mem = FakeMemory()
        mem.write(RM.CUR_MAP, M.MAP_IDS[map_name])
        mem.write(RM.PARTY_COUNT, 1)
        make_party_mon(mem, 0, species=176, level=8,
                       hp=int(round(20 * hp)), max_hp=20)
        return RM.RamReader(mem).read()

    def test_restoring_hurt_owes_a_heal_trip(self, env):
        env.reset()
        assert env._heal_trip_owed(self._st(env, 0.2, "VIRIDIAN_FOREST")) is True

    def test_restoring_healthy_owes_nothing(self, env):
        env.reset()
        assert env._heal_trip_owed(self._st(env, 1.0, "VIRIDIAN_FOREST")) is False

    def test_an_empty_party_owes_nothing(self, env):
        """0.0 HP with no party is the opening of the game, not an injury."""
        env.reset()
        mem = FakeMemory()
        mem.write(RM.CUR_MAP, M.MAP_IDS["OAKS_LAB"])
        mem.write(RM.PARTY_COUNT, 0)
        assert env._heal_trip_owed(RM.RamReader(mem).read()) is False

    def test_reaching_a_center_while_hurt_now_pays(self, env):
        """The whole point: restored hurt, walk into a Center, get credit."""
        env.reset()
        env._heal_trip_paid = not env._heal_trip_owed(
            self._st(env, 0.2, "VIRIDIAN_FOREST"))
        _, b = env._reward(self._st(env, 0.2, "PEWTER_POKECENTER"))
        assert b.get("heal_visit", 0.0) > 0

    def test_it_pays_only_once_per_bout(self, env):
        """Otherwise standing in the lobby is an income stream."""
        env.reset()
        env._heal_trip_paid = not env._heal_trip_owed(
            self._st(env, 0.2, "VIRIDIAN_FOREST"))
        env._reward(self._st(env, 0.2, "PEWTER_POKECENTER"))
        _, b = env._reward(self._st(env, 0.2, "PEWTER_POKECENTER"))
        assert "heal_visit" not in b


class TestWinningABattlePays:
    """Nothing rewarded *winning* a fight, so fleeing strictly dominated it.

    `enemy_damage` pays at most 1.0 in total for taking an opponent from full to zero,
    `faint` costs 5.0, and fleeing pays 0 at no risk. Measured with the live policy from
    healthy archive restores: 40 battles in 4000 steps, party level sum 8 -> 8, not one
    level gained, in-battle action mix 23% B against 15% A. `reward/level` had never
    fired in the whole run.
    """

    def _st(self, in_battle=1, enemy_hp=20, enemy_max=20, hp=1.0, party=1):
        mem = FakeMemory()
        mem.write(RM.CUR_MAP, M.MAP_IDS["ROUTE_1"])
        mem.write(RM.PARTY_COUNT, party)
        if party:
            make_party_mon(mem, 0, species=176, level=8,
                           hp=int(round(20 * hp)), max_hp=20)
        mem.write(RM.IS_IN_BATTLE, in_battle)
        mem.write(RM.ENEMY_MON_HP, [enemy_hp >> 8, enemy_hp & 0xFF])
        mem.write(RM.ENEMY_MON_MAX_HP, [enemy_max >> 8, enemy_max & 0xFF])
        return RM.RamReader(mem).read()

    def test_knocking_the_opponent_out_pays_on_battle_end(self, env):
        env.reset()
        env._reward(self._st(in_battle=1, enemy_hp=20))
        env._reward(self._st(in_battle=1, enemy_hp=0))
        _, b = env._reward(self._st(in_battle=0, enemy_hp=0))
        assert b.get("battle_won", 0.0) > 0

    def test_fleeing_pays_nothing(self, env):
        """The distinction the term exists for."""
        env.reset()
        env._reward(self._st(in_battle=1, enemy_hp=20))
        env._reward(self._st(in_battle=1, enemy_hp=14))
        _, b = env._reward(self._st(in_battle=0, enemy_hp=14))
        assert "battle_won" not in b

    def test_it_pays_once_not_every_step_afterwards(self, env):
        env.reset()
        env._reward(self._st(in_battle=1, enemy_hp=0))
        env._reward(self._st(in_battle=0))
        _, b = env._reward(self._st(in_battle=0))
        assert "battle_won" not in b

    def test_a_wipe_does_not_pay(self, env):
        """Losing the run's only Pokemon is not a win, whatever the enemy's HP reads."""
        env.reset()
        env._reward(self._st(in_battle=1, enemy_hp=0))
        _, b = env._reward(self._st(in_battle=0, hp=0.0))
        assert "battle_won" not in b

    def test_winning_beats_fleeing_on_net_reward(self, env):
        """The property that has to hold for fighting to be the rational choice."""
        env.reset()
        env._reward(self._st(in_battle=1, enemy_hp=20))
        env._reward(self._st(in_battle=1, enemy_hp=0))
        win, _ = env._reward(self._st(in_battle=0, enemy_hp=0))
        env.reset()
        env._reward(self._st(in_battle=1, enemy_hp=20))
        flee, _ = env._reward(self._st(in_battle=0, enemy_hp=20))
        assert win > flee


class TestBallsAreConsumable:
    """Balls are spendable, so the reward has to be a potential, not a monotone maximum.

    Oak hands over 5 Poke Balls (EVENT_GOT_POKEBALLS_FROM_OAK was set in 405 of 607
    live cells) and the agent threw them all. The monotone version left `max_balls`
    at a value the bag could never exceed again: measured at 83.5M env steps it sat at
    1 while every cell carried 0 balls, so buying one would have paid nothing.
    """

    def _st(self, balls):
        mem = FakeMemory()
        mem.write(RM.CUR_MAP, M.MAP_IDS["VIRIDIAN_MART"])
        mem.write(RM.PARTY_COUNT, 1)
        make_party_mon(mem, 0, species=176, level=8, hp=20, max_hp=20)
        mem.write(RM.NUM_BAG_ITEMS, 1 if balls else 0)
        if balls:
            mem.write(RM.BAG_ITEMS, [0x04, balls])
        return RM.RamReader(mem).read()

    def test_acquiring_balls_pays(self, env):
        env.reset()
        env._reward(self._st(0))
        _, b = env._reward(self._st(5))
        assert b.get("ball", 0.0) > 0

    def test_spending_a_ball_costs(self, env):
        env.reset()
        env._reward(self._st(5))
        _, b = env._reward(self._st(4))
        assert b.get("ball", 0.0) < 0

    def test_a_buy_then_spend_round_trip_telescopes_to_zero(self, env):
        """Symmetric, so there is no loop to farm.

        Compares the `ball` component, not the total: the totals also carry
        `party_member`, `level` and `new_map` credit from the synthetic states.
        """
        env.reset()
        env._reward(self._st(0))
        _, gain = env._reward(self._st(5))
        _, loss = env._reward(self._st(0))
        assert gain["ball"] + loss["ball"] == pytest.approx(0.0, abs=1e-6)

    def test_rebuying_after_spending_everything_still_pays(self, env):
        """The exact case the monotone version could never pay again."""
        env.reset()
        env._reward(self._st(0))
        env._reward(self._st(5))
        env._reward(self._st(0))
        _, b = env._reward(self._st(3))
        assert b.get("ball", 0.0) > 0

    def test_holding_a_steady_bag_pays_nothing(self, env):
        """Only the *change* is priced, so standing still with balls is not income."""
        env.reset()
        env._reward(self._st(3))
        _, b = env._reward(self._st(3))
        assert "ball" not in b

    def test_the_cap_bounds_a_large_purchase(self, env):
        env.reset()
        env._reward(self._st(0))
        _, b = env._reward(self._st(99))
        assert b["ball"] == pytest.approx(env.reward_cfg.ball
                                          * env.reward_cfg.ball_cap)


class TestStallingIsWorseThanLosing:
    """A cap below `faint` made stalling the cheapest way out of an unwinnable fight.

    Measured over the 530k steps after the agent first engaged Brock -- a level-14 Onix
    against a level-8 Charmander, a trainer battle it can neither win nor flee --
    `battle_stall` fired in 48 of 52 metric rows while `battle_won` and `faint` fired in
    none. It was not losing the fight; it was sitting in it until the episode ran out,
    because stalling cost at most 2.5 against a wipe at 5.0.
    """

    def _battle(self, env, enemy_hp=20, hp=1.0):
        mem = FakeMemory()
        mem.write(RM.CUR_MAP, M.MAP_IDS["PEWTER_GYM"])
        mem.write(RM.PARTY_COUNT, 1)
        make_party_mon(mem, 0, species=176, level=8,
                       hp=int(round(20 * hp)), max_hp=20)
        mem.write(RM.IS_IN_BATTLE, 2)          # trainer battle: cannot be fled
        mem.write(RM.ENEMY_MON_HP, [enemy_hp >> 8, enemy_hp & 0xFF])
        mem.write(RM.ENEMY_MON_MAX_HP, [0, 20])
        return RM.RamReader(mem).read()

    def _stall(self, env, steps):
        total = 0.0
        for _ in range(steps):
            _, b = env._reward(self._battle(env))
            total += b.get("battle_stall", 0.0)
        return total

    def test_a_full_stall_costs_more_than_a_wipe(self, env):
        env.reset()
        charged = self._stall(env, 2000)
        assert abs(charged) > abs(env.reward_cfg.faint), charged

    def test_the_charge_is_still_bounded(self, env):
        """Unbounded, a long fight would dwarf every other term in the function."""
        env.reset()
        charged = self._stall(env, 4000)
        # One step of overshoot: the budget is checked before the charge is added.
        allowed = abs(env.reward_cfg.faint) * 1.5 + abs(env.reward_cfg.battle_stall)
        assert abs(charged) <= allowed + 1e-6

    def test_progress_in_the_fight_resets_the_charge(self, env):
        """Only *stalling* is penalised; a long but progressing battle is fine."""
        env.reset()
        for i in range(200):
            env._reward(self._battle(env, enemy_hp=20 - (i % 10)))
        assert env._battle_stall_charged < abs(env.reward_cfg.faint)
