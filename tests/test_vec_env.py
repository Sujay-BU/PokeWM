"""Worker -> trainer payload.

`state_blob` is captured once, when a cell key first appears, and rides along unchanged
on every later step of the episode. Anything the archive stores *about* that state has
to be captured with it, or the archive describes a state it does not hold.
"""

from __future__ import annotations

import pytest

from pokewm.agent.milestones import MilestoneTracker
from pokewm.emulator import maps as M
from pokewm.emulator import ram_map as RM
from pokewm.emulator.vec_env import _pack_info

from .conftest import FakeMemory, make_party_mon


def gs_with(hp_frac: float, level: int):
    """A real GameState, so the payload is exercised against the actual fields."""
    mem = FakeMemory()
    mem.write(RM.CUR_MAP, M.MAP_IDS["VIRIDIAN_FOREST"])
    mem.write(RM.X_COORD, 5)
    mem.write(RM.Y_COORD, 5)
    mem.write(RM.PARTY_COUNT, 1)
    make_party_mon(mem, 0, species=176, level=level,
                   hp=int(round(20 * hp_frac)), max_hp=20)
    return RM.RamReader(mem).read()


def pack(gs, blob_hp, blob_lvl, blob=b"blob"):
    return _pack_info({"state": gs}, MilestoneTracker(), blob,
                      blob_hp_frac=blob_hp, blob_level_sum=blob_lvl)


class TestSnapshotMetadataMatchesTheSnapshot:
    """Measured on a live archive at 57.9M env steps: recorded `hp_frac` took exactly
    two values across 374 cells -- 1.0 for the 242 the trainer inserted (its default,
    because the key was absent from this payload) and 0.0 for the 132 inherited from an
    older archive. Health was never measured, so `hp_weight` and `frontier_min_hp`
    ranked cells by a constant, and `--from-frontier` handed the viewer a "healthy"
    Pewter cell that restored at 0.42 HP and blacked out 111 steps later.
    """

    def test_health_of_the_snapshot_is_carried(self):
        packed = pack(gs_with(hp_frac=0.1, level=20), blob_hp=0.9, blob_lvl=6)
        assert packed["blob_hp_frac"] == pytest.approx(0.9)

    def test_health_is_not_silently_defaulted(self):
        """The original bug: no health key at all, so the caller's default always won."""
        packed = pack(gs_with(hp_frac=0.25, level=6), blob_hp=0.25, blob_lvl=6)
        assert "blob_hp_frac" in packed
        assert packed["blob_hp_frac"] != 1.0

    def test_snapshot_level_differs_from_current_level(self):
        """`level_sum` is read fresh each step; the blob is not."""
        packed = pack(gs_with(hp_frac=1.0, level=20), blob_hp=1.0, blob_lvl=6)
        assert packed["blob_level_sum"] == 6
        assert packed["level_sum"] == 20

    def test_the_two_agree_when_the_snapshot_is_the_current_step(self):
        gs = gs_with(hp_frac=0.55, level=11)
        packed = pack(gs, blob_hp=gs.party_hp_frac, blob_lvl=gs.party_level_sum)
        assert packed["blob_hp_frac"] == pytest.approx(gs.party_hp_frac)
        assert packed["blob_level_sum"] == packed["level_sum"]

    def test_a_blobless_step_still_packs(self):
        """Most steps carry no snapshot; the payload must not require one."""
        packed = pack(gs_with(1.0, 6), blob_hp=1.0, blob_lvl=6, blob=None)
        assert packed["state_blob"] is None


class TestAWipedPartyIsNotArchived:
    """Restoring a cell whose party is at zero HP blacks out on the next step and
    teleports to `wLastBlackoutMap`, so the episode is spent somewhere the archive did
    not choose. Measured: 9 of 378 live cells, 7 of them in Oak's Lab.

    The predicate mirrors `_worker`'s `alive` check; the worker itself needs a live
    emulator, so the condition is pinned here against real GameStates.
    """

    @staticmethod
    def alive(gs):
        return gs.party_size == 0 or gs.party_hp_frac > 0.0

    def test_a_wiped_party_is_rejected(self):
        assert not self.alive(gs_with(hp_frac=0.0, level=6))

    def test_a_hurt_but_standing_party_is_kept(self):
        """Low HP is a ranking matter for `hp_weight`, not a reason to discard."""
        assert self.alive(gs_with(hp_frac=0.05, level=6))

    def test_a_healthy_party_is_kept(self):
        assert self.alive(gs_with(hp_frac=1.0, level=6))

    def test_an_empty_party_is_kept(self):
        """The opening of the game reads 0.0 HP with no party. It is not a wipe --
        76 of 378 live cells are pre-starter and all of them are legitimate."""
        mem = FakeMemory()
        mem.write(RM.CUR_MAP, M.MAP_IDS["OAKS_LAB"])
        mem.write(RM.PARTY_COUNT, 0)
        gs = RM.RamReader(mem).read()
        assert gs.party_size == 0 and gs.party_hp_frac == 0.0
        assert self.alive(gs)


class TestSnapshotsFollowImprovement:
    """A snapshot taken once per cell key per episode cannot capture grinding.

    `progress_key` hashes badges, dex, party size and story flags. A wild-battle win
    changes none of them, so the key is fixed for the episode and the stored state stays
    as it was around step 2. Measured at 86.4M env steps: `battle_won` firing in 145 of
    146 metric rows, no archived cell above 327 XP, `level_sum` pinned at 8.

    The worker owns a live emulator, so the trigger is pinned here as a predicate.
    """

    @staticmethod
    def should_snapshot(key, reported, gs_exp, gs_hp, blob_exp, blob_hp,
                        controllable=True):
        stronger = gs_exp > blob_exp
        healthier = gs_hp > blob_hp + 0.05
        return controllable and (key not in reported or stronger or healthier)

    def test_a_new_key_snapshots(self):
        assert self.should_snapshot("k", set(), 327, 1.0, 0, 1.0)

    def test_a_repeated_key_with_nothing_gained_does_not(self):
        assert not self.should_snapshot("k", {"k"}, 327, 1.0, 327, 1.0)

    def test_experience_gained_under_the_same_key_snapshots(self):
        """The case that was losing every wild-battle win."""
        assert self.should_snapshot("k", {"k"}, 380, 1.0, 327, 1.0)

    def test_healing_under_the_same_key_snapshots(self):
        assert self.should_snapshot("k", {"k"}, 327, 0.95, 327, 0.20)

    def test_taking_damage_does_not_snapshot(self):
        """Only improvement is worth re-storing; a hurt state is not a better launch pad."""
        assert not self.should_snapshot("k", {"k"}, 327, 0.20, 327, 1.0)

    def test_an_uncontrollable_state_never_snapshots(self):
        """Improvement does not override the mid-warp / mid-battle guards."""
        assert not self.should_snapshot("k", {"k"}, 999, 1.0, 327, 0.1,
                                        controllable=False)
