"""Frontier archive.

The proof's assumption A2 is that the frontier never regresses: once a cell at milestone
m is archived, the archive's maximum stays >= m for the rest of the run. Eviction is the
only thing that could break that, so it gets the most attention here.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pokewm.config import ArchiveConfig
from pokewm.emulator import maps as M
from pokewm.emulator.archive import Cell, FrontierArchive


def add(arch: FrontierArchive, key: str, milestone: int, seen=None, blob=None,
        map_id=0, badges=0, events=0) -> bool:
    return arch.insert(
        key=key,
        blob=blob if blob is not None else key.encode(),
        milestone=milestone,
        map_id=map_id,
        badges=badges,
        events=events,
        seen_maps=frozenset(seen or {map_id}),
    )


@pytest.fixture
def arch() -> FrontierArchive:
    return FrontierArchive(ArchiveConfig(max_cells=8, restore_prob=1.0), seed=0)


class TestInsertion:
    def test_first_insert_is_new(self, arch):
        assert add(arch, "a", 3) is True
        assert len(arch) == 1

    def test_duplicate_key_is_not_new(self, arch):
        add(arch, "a", 3)
        assert add(arch, "a", 3) is False
        assert len(arch) == 1

    def test_revisits_increment_the_visit_count(self, arch):
        add(arch, "a", 3)
        add(arch, "a", 3)
        add(arch, "a", 3)
        assert arch.deepest().visits == 3

    def test_deeper_representative_replaces_the_blob(self, arch):
        add(arch, "a", 3, blob=b"shallow")
        add(arch, "a", 5, blob=b"deeper")
        cell = arch.deepest()
        assert cell.milestone == 5 and cell.blob == b"deeper"

    def test_same_depth_more_exploration_replaces_the_blob(self, arch):
        add(arch, "a", 3, seen={1}, blob=b"narrow")
        add(arch, "a", 3, seen={1, 2, 3}, blob=b"wide")
        assert arch.deepest().blob == b"wide"

    def test_shallower_does_not_replace(self, arch):
        add(arch, "a", 5, blob=b"good")
        add(arch, "a", 2, blob=b"bad")
        assert arch.deepest().blob == b"good"
        assert arch.deepest().milestone == 5

    def test_max_milestone_tracks_the_deepest_insert(self, arch):
        add(arch, "a", 2)
        add(arch, "b", 9)
        add(arch, "c", 4)
        assert arch.max_milestone == 9


class TestEviction:
    def test_respects_capacity(self, arch):
        for i in range(30):
            add(arch, f"k{i}", milestone=i % 4)
        assert len(arch) <= 8

    def test_never_evicts_the_unique_deepest_cell(self):
        a = FrontierArchive(ArchiveConfig(max_cells=4), seed=0)
        add(a, "deep", 99)
        for i in range(50):
            add(a, f"shallow{i}", 1)
        assert a.deepest().milestone == 99
        assert "deep" in [c.key for c in a._cells.values()]

    def test_frontier_never_regresses_under_pressure(self):
        """Assumption A2 of docs/PROOF.md, exercised directly."""
        a = FrontierArchive(ArchiveConfig(max_cells=6), seed=0)
        rng = np.random.default_rng(0)
        best = 0
        for i in range(500):
            m = int(rng.integers(0, 30))
            add(a, f"k{i}", m)
            best = max(best, m)
            assert a.deepest().milestone == best, f"regressed at insert {i}"

    def test_evicts_shallowest_first(self):
        a = FrontierArchive(ArchiveConfig(max_cells=3), seed=0)
        add(a, "m1", 1)
        add(a, "m5", 5)
        add(a, "m9", 9)
        add(a, "m7", 7)  # forces one eviction
        keys = {c.key for c in a._cells.values()}
        assert "m1" not in keys
        assert {"m5", "m9", "m7"} <= keys | {"m7"}


class TestPerLevelCap:
    """Regression: shallow levels crowded out the frontier.

    A measured run held 83 Pallet Town cells against 11 on Route 1, so only ~3% of
    restores landed on the frontier (docs/PROOF.md §3.2 bounds sigma by n_0/n_max).
    """

    def test_superseded_level_population_is_capped(self):
        a = FrontierArchive(ArchiveConfig(max_cells=10_000, max_cells_per_level=8), seed=0)
        for i in range(60):
            add(a, f"shallow{i}", milestone=2)
        # While level 2 is the frontier it is exempt.
        assert len([c for c in a._cells.values() if c.milestone == 2]) == 60
        # Once a deeper level exists it is superseded and gets trimmed.
        add(a, "deeper", milestone=5)
        at2 = [c for c in a._cells.values() if c.milestone == 2]
        assert len(at2) <= 8, len(at2)

    def test_cap_is_per_level_not_global(self):
        a = FrontierArchive(ArchiveConfig(max_cells=10_000, max_cells_per_level=4), seed=0)
        for lvl in range(5):
            for i in range(20):
                add(a, f"l{lvl}c{i}", milestone=lvl)
        for lvl in range(4):  # levels 0..3 are superseded by level 4
            assert len([c for c in a._cells.values() if c.milestone == lvl]) <= 4
        assert len([c for c in a._cells.values() if c.milestone == 4]) == 20

    def test_frontier_fraction_stays_healthy(self):
        """The property the cap exists for: n_0/n_max must not collapse."""
        a = FrontierArchive(ArchiveConfig(max_cells=10_000, max_cells_per_level=32), seed=0)
        for i in range(200):          # lots of shallow wandering
            add(a, f"shallow{i}", milestone=2)
        for i in range(11):           # a thin frontier, as actually measured
            add(a, f"deep{i}", milestone=6)
        # The shallow backlog must be trimmed retroactively once level 6 appears --
        # it was inserted while level 2 was still the frontier and therefore exempt.
        import collections
        pops = collections.Counter(c.milestone for c in a._cells.values())
        n0, nmax = pops[6], max(pops.values())
        assert n0 / nmax > 0.3, f"frontier starved: {n0}/{nmax}"

    def test_frontier_level_is_exempt_from_the_cap(self):
        """The frontier needs spatial coverage; capping it blocks the ratchet.

        Worse, the cap evicts the most-chosen cells, which at the frontier are exactly
        the ones being used to push forward.
        """
        a = FrontierArchive(ArchiveConfig(max_cells=10_000, max_cells_per_level=4), seed=0)
        for i in range(40):
            add(a, f"deep{i}", milestone=7)
        at7 = [c for c in a._cells.values() if c.milestone == 7]
        assert len(at7) == 40, f"frontier was capped to {len(at7)}"

    def test_shallow_levels_are_still_capped_once_a_deeper_one_exists(self):
        a = FrontierArchive(ArchiveConfig(max_cells=10_000, max_cells_per_level=4), seed=0)
        add(a, "deep", milestone=9)
        for i in range(40):
            add(a, f"shallow{i}", milestone=2)
        assert len([c for c in a._cells.values() if c.milestone == 2]) <= 4
        assert a.deepest().milestone == 9

    def test_deepest_cell_survives_the_cap(self):
        a = FrontierArchive(ArchiveConfig(max_cells=10_000, max_cells_per_level=2), seed=0)
        add(a, "unique_deep", milestone=9)
        for i in range(30):
            add(a, f"s{i}", milestone=1)
        assert a.deepest().milestone == 9

    def test_disabled_when_zero(self):
        a = FrontierArchive(ArchiveConfig(max_cells=10_000, max_cells_per_level=0), seed=0)
        for i in range(40):
            add(a, f"c{i}", milestone=3)
        assert len(a) == 40


class TestSelection:
    def test_returns_none_when_empty(self, arch):
        assert arch.sample() is None

    def test_returns_none_when_restore_prob_is_zero(self):
        a = FrontierArchive(ArchiveConfig(restore_prob=0.0), seed=0)
        add(a, "a", 1)
        assert all(a.sample() is None for _ in range(20))

    def test_prefers_deeper_cells(self):
        a = FrontierArchive(ArchiveConfig(restore_prob=1.0, temperature=1.0,
                                          novelty_weight=0.0), seed=0)
        add(a, "shallow", 1)
        add(a, "deep", 10)
        picks = [a.sample().key for _ in range(200)]
        assert picks.count("deep") > picks.count("shallow") * 5

    def test_novelty_bonus_spreads_selection(self):
        """Count-based bonus must stop the archive collapsing onto one dead-end cell."""
        a = FrontierArchive(
            ArchiveConfig(restore_prob=1.0, temperature=1.0, novelty_weight=5.0), seed=0
        )
        add(a, "deep", 10)
        for i in range(5):
            add(a, f"peer{i}", 9)
        picks = [a.sample().key for _ in range(300)]
        assert len(set(picks)) >= 4, "selection collapsed onto a single cell"

    def test_chosen_counter_increments(self, arch):
        add(arch, "a", 1)
        for _ in range(7):
            arch.sample()
        assert arch.deepest().chosen == 7

    def test_score_decreases_with_repeated_selection(self):
        a = FrontierArchive(ArchiveConfig(novelty_weight=2.0), seed=0)
        add(a, "a", 5)
        cell = a.deepest()
        first = cell.score(2.0)
        cell.chosen += 20
        assert cell.score(2.0) < first


class TestRelabel:
    """Regression: stale milestone labels survived a chain change.

    A run carried 242 cells claiming milestone 8 under a chain where those states were
    milestone 7. Wrong labels are not cosmetic: they outrank everything and the deepest
    level is exempt from trimming, so they never age out.
    """

    def test_recomputes_milestones_and_rewrites_keys(self):
        a = FrontierArchive(ArchiveConfig(max_cells=100, max_cells_per_level=100), seed=0)
        for i in range(5):
            a.insert(key=f"hash{i}:8:12:0:0", blob=b"x", milestone=8, map_id=12,
                     badges=0, events=0, seen_maps=frozenset({12}))
        assert a.max_milestone == 8
        changed = a.relabel(lambda c: 7)
        assert changed == 5
        assert a.max_milestone == 7
        assert all(c.milestone == 7 for c in a._cells.values())
        # the key embeds the milestone and must be rewritten with it
        assert all(c.key.split(":")[1] == "7" for c in a._cells.values())

    def test_unchanged_cells_are_left_alone(self):
        a = FrontierArchive(ArchiveConfig(max_cells=100, max_cells_per_level=100), seed=0)
        a.insert(key="h:5:12:0:0", blob=b"x", milestone=5, map_id=12, badges=0,
                 events=0, seen_maps=frozenset({12}))
        assert a.relabel(lambda c: 5) == 0
        assert a.deepest().key == "h:5:12:0:0"

    def test_a_corrupt_cell_does_not_destroy_the_archive(self):
        a = FrontierArchive(ArchiveConfig(max_cells=100, max_cells_per_level=100), seed=0)
        for i in range(4):
            a.insert(key=f"h{i}:8:12:0:0", blob=b"x", milestone=8, map_id=12,
                     badges=0, events=0, seen_maps=frozenset({12}))

        def flaky(cell):
            if cell.key.startswith("h2"):
                raise RuntimeError("corrupt save state")
            return 6

        a.relabel(flaky)
        assert len(a) == 3, "only the unreadable cell should be dropped"
        assert all(c.milestone == 6 for c in a._cells.values())

    def test_relabel_reapplies_level_caps(self):
        a = FrontierArchive(ArchiveConfig(max_cells=1000, max_cells_per_level=4), seed=0)
        a.insert(key="deep:9:12:0:0", blob=b"x", milestone=9, map_id=12, badges=0,
                 events=0, seen_maps=frozenset({12}))
        for i in range(30):
            a.insert(key=f"h{i}:8:12:0:0", blob=b"x", milestone=8, map_id=12,
                     badges=0, events=0, seen_maps=frozenset({12}))
        # collapse everything to one shallow level; the cap must then bite
        a.relabel(lambda c: 9 if c.key.startswith("deep") else 2)
        assert len([c for c in a._cells.values() if c.milestone == 2]) <= 4


class TestMapRankScoring:
    """Regression: backtracked cells outnumbered real frontier cells 50 to 12."""

    def test_forward_cell_outscores_a_backtracked_one_at_the_same_milestone(self):
        from pokewm.emulator import maps as M

        cfg = ArchiveConfig(novelty_weight=1.0, map_rank_weight=1.0)
        a = FrontierArchive(cfg, seed=0)
        add(a, "onroute", 6, map_id=M.MAP_IDS["ROUTE_1"])
        add(a, "backtracked", 6, map_id=M.MAP_IDS["REDS_HOUSE_1F"])
        cells = {c.key: c for c in a._cells.values()}
        s_fwd = cells["onroute"].score(cfg.novelty_weight, cfg.map_rank_weight)
        s_back = cells["backtracked"].score(cfg.novelty_weight, cfg.map_rank_weight)
        assert s_fwd > s_back, (s_fwd, s_back)

    def test_selection_strongly_prefers_forward_cells(self):
        from pokewm.emulator import maps as M

        a = FrontierArchive(
            ArchiveConfig(restore_prob=1.0, novelty_weight=1.0, map_rank_weight=1.0,
                          max_cells_per_level=10_000),
            seed=0,
        )
        for i in range(12):
            add(a, f"route{i}", 6, map_id=M.MAP_IDS["ROUTE_1"])
        for i in range(50):  # the measured backtracked majority
            add(a, f"back{i}", 6, map_id=M.MAP_IDS["PALLET_TOWN"])
        picks = [a.sample().key for _ in range(300)]
        fwd = sum(1 for k in picks if k.startswith("route"))
        assert fwd / len(picks) > 0.7, (
            f"only {fwd}/{len(picks)} restores landed on the forward cells despite "
            "being outnumbered 50 to 12"
        )

    def test_disabled_when_weight_is_zero(self):
        from pokewm.emulator import maps as M

        cfg = ArchiveConfig(novelty_weight=0.0, map_rank_weight=0.0)
        a = FrontierArchive(cfg, seed=0)
        add(a, "onroute", 6, map_id=M.MAP_IDS["ROUTE_1"])
        add(a, "backtracked", 6, map_id=M.MAP_IDS["REDS_HOUSE_1F"])
        cells = {c.key: c for c in a._cells.values()}
        assert cells["onroute"].score(0.0, 0.0) == cells["backtracked"].score(0.0, 0.0)


class TestEvictionPreservesProgress:
    """Regression: the archive deleted its own launch pad.

    Evicting the most-chosen cell is self-destructive under a hard cap -- the cells the
    archive relies on are the ones it picks most, so they accrue the highest `chosen`
    count and go first. A live run lost *every* Viridian City cell that way and fell back
    to Route 1, a map further behind.
    """

    def test_advanced_cells_survive_even_when_heavily_chosen(self):
        from pokewm.emulator import maps as M

        a = FrontierArchive(
            ArchiveConfig(max_cells=10_000, max_cells_per_level=4), seed=0
        )
        add(a, "deeper", 9, map_id=M.MAP_IDS["ROUTE_1"])  # keeps level 5 non-frontier
        # One heavily-used far cell, plus filler close to the start.
        add(a, "viridian", 5, map_id=M.MAP_IDS["VIRIDIAN_CITY"])
        a._cells["viridian"].chosen = 500
        for i in range(20):
            add(a, f"pallet{i}", 5, map_id=M.MAP_IDS["PALLET_TOWN"])

        keys = {c.key for c in a._cells.values()}
        assert "viridian" in keys, "the most-advanced cell was evicted for being useful"
        assert len([c for c in a._cells.values() if c.milestone == 5]) <= 4

    def test_least_advanced_is_evicted_first(self):
        from pokewm.emulator import maps as M

        a = FrontierArchive(
            ArchiveConfig(max_cells=10_000, max_cells_per_level=2), seed=0
        )
        add(a, "deeper", 9, map_id=M.MAP_IDS["ROUTE_1"])
        add(a, "far", 5, map_id=M.MAP_IDS["VIRIDIAN_CITY"])   # rank 6
        add(a, "mid", 5, map_id=M.MAP_IDS["ROUTE_1"])         # rank 5
        add(a, "near", 5, map_id=M.MAP_IDS["PALLET_TOWN"])    # rank 1
        keys = {c.key for c in a._cells.values()}
        assert "near" not in keys
        assert {"far", "mid"} <= keys

    def test_chosen_still_breaks_ties_between_equally_advanced_cells(self):
        """Among cells on the *same* map the old heuristic is right.

        Spatially equivalent cells carry no extra progress, so the most-exploited one is
        the fair victim. The fix was only to stop that overriding spatial progress.
        """
        from pokewm.emulator import maps as M

        a = FrontierArchive(
            ArchiveConfig(max_cells=10_000, max_cells_per_level=3), seed=0
        )
        add(a, "deeper", 9, map_id=M.MAP_IDS["ROUTE_1"])
        for name in ("a", "b", "c"):
            add(a, name, 5, map_id=M.MAP_IDS["PALLET_TOWN"])
        a._cells["a"].chosen = 100
        a._cells["b"].chosen = 1
        a._cells["c"].chosen = 0
        add(a, "d", 5, map_id=M.MAP_IDS["PALLET_TOWN"])  # forces one eviction
        keys = {c.key for c in a._cells.values()}
        assert "a" not in keys, "the most-exploited equivalent cell should go first"
        assert {"b", "c", "d"} <= keys


class TestTargetFallback:
    """Regression: 'reach map X' milestones targeted a map with no cells.

    The target is by definition somewhere the agent has never been, so the bonus applied
    to nothing -- measured 0.0% on-target, restores scattering 46.6% into Oak's Lab.
    """

    def _archive(self):
        from pokewm.emulator import maps as M

        a = FrontierArchive(
            ArchiveConfig(restore_prob=1.0, target_bonus=6.0, map_rank_weight=1.0,
                          max_cells_per_level=10_000),
            seed=0,
        )
        # All at the same milestone on purpose: the fallback rule under test picks a
        # *map*, and mixing levels here would entangle it with the frontier rule that
        # `TestFrontierRestore` covers.
        for i in range(10):
            add(a, f"pallet{i}", 9, map_id=M.MAP_IDS["PALLET_TOWN"])
        for i in range(10):
            add(a, f"lab{i}", 9, map_id=M.MAP_IDS["OAKS_LAB"])
        for i in range(6):
            add(a, f"viridian{i}", 9, map_id=M.MAP_IDS["VIRIDIAN_CITY"])
        return a, M

    def test_unreached_target_falls_back_to_the_deepest_reached_map(self):
        a, M = self._archive()
        a.set_target_maps({M.MAP_IDS["ROUTE_2"]})  # never visited
        # Viridian City is the furthest along the critical path that exists here.
        assert a._target_maps == {M.MAP_IDS["VIRIDIAN_CITY"]}

    def test_fallback_actually_redirects_restores(self):
        import collections

        a, M = self._archive()
        a.set_target_maps({M.MAP_IDS["ROUTE_2"]})
        picks = collections.Counter()
        for _ in range(600):
            c = a.sample()
            if c:
                picks[c.map_id] += 1
        frac = picks[M.MAP_IDS["VIRIDIAN_CITY"]] / max(sum(picks.values()), 1)
        assert frac > 0.6, f"only {frac:.1%} of restores reached the frontier map"

    def test_a_reachable_target_is_used_as_given(self):
        a, M = self._archive()
        a.set_target_maps({M.MAP_IDS["OAKS_LAB"]})
        assert a._target_maps == {M.MAP_IDS["OAKS_LAB"]}

    def test_empty_archive_keeps_the_request(self):
        from pokewm.emulator import maps as M

        a = FrontierArchive(ArchiveConfig(), seed=0)
        a.set_target_maps({M.MAP_IDS["ROUTE_2"]})
        assert a._target_maps == {M.MAP_IDS["ROUTE_2"]}

    def test_off_path_only_archive_does_not_crash(self):
        from pokewm.emulator import maps as M

        a = FrontierArchive(ArchiveConfig(), seed=0)
        add(a, "mart", 3, map_id=M.MAP_IDS["VIRIDIAN_MART"])  # map_rank -1
        a.set_target_maps({M.MAP_IDS["ROUTE_2"]})
        assert a.sample() is not None


class TestFrontierRestore:
    """Regression: restores launched into a world where the objective was sealed off.

    After Oak's Parcel was delivered (milestone 9), 22 of the 24 archived Route 1 cells
    were milestone-8 states -- parcel picked up, *not* delivered. In those the old man
    still blocks Viridian's north exit, so Route 2 cannot be entered at all. The cell
    score weighted milestone at 1.0 against map_rank + target_bonus summing to 11, so
    the 22 stale cells outvoted the 2 usable ones and ~80% of on-target restores began
    in an unwinnable world. The run held milestone 9 for 2.0M env steps while a
    map-only diagnostic reported 99.8% on-target.

    Milestone is not merely "further along": it encodes irreversible world state.
    """

    def _archive(self, **kw):
        from pokewm.emulator import maps as M

        cfg = ArchiveConfig(
            restore_prob=1.0, target_bonus=6.0, map_rank_weight=1.0,
            max_cells_per_level=10_000, frontier_min_cells=1, **kw,
        )
        a = FrontierArchive(cfg, seed=0)
        for i in range(22):
            add(a, f"stale{i}", 8, map_id=M.MAP_IDS["ROUTE_1"])
        for i in range(2):
            add(a, f"live{i}", 9, map_id=M.MAP_IDS["ROUTE_1"])
        return a, M

    def _sample_levels(self, a, n=1000):
        import collections

        picks = collections.Counter()
        for _ in range(n):
            c = a.sample()
            if c:
                picks[c.milestone] += 1
        return picks, max(sum(picks.values()), 1)

    def test_frontier_level_dominates_a_more_numerous_stale_level(self):
        a, M = self._archive()
        a.set_target_maps({M.MAP_IDS["ROUTE_2"]})
        picks, total = self._sample_levels(a)
        assert picks[9] / total > 0.7, f"only {picks[9] / total:.1%} launched post-gate"

    def test_stale_levels_are_not_shut_out_entirely(self):
        """A frontier that turns out to be a dead end has to be escapable."""
        a, M = self._archive()
        a.set_target_maps({M.MAP_IDS["ROUTE_2"]})
        picks, total = self._sample_levels(a)
        assert picks[8] > 0

    def test_disabling_the_rule_restores_the_old_behaviour(self):
        a, M = self._archive(frontier_prob=0.0)
        a.set_target_maps({M.MAP_IDS["ROUTE_2"]})
        picks, total = self._sample_levels(a)
        assert picks[8] / total > 0.5

    def test_a_thin_frontier_widens_to_the_level_below(self):
        """One cell at a new level must not become the only launch pad."""
        from pokewm.emulator import maps as M

        a = FrontierArchive(
            ArchiveConfig(restore_prob=1.0, frontier_prob=1.0, frontier_min_cells=6,
                          max_cells_per_level=10_000),
            seed=0,
        )
        for i in range(8):
            add(a, f"below{i}", 9, map_id=M.MAP_IDS["ROUTE_1"])
        add(a, "tip", 10, map_id=M.MAP_IDS["ROUTE_2"])
        picks, total = self._sample_levels(a, n=300)
        assert picks[10] > 0 and picks[9] > 0

    def test_target_fallback_ignores_maps_only_stale_levels_reach(self):
        """The bonus must not point at a map the frontier cannot supply a cell for."""
        from pokewm.emulator import maps as M

        a = FrontierArchive(
            ArchiveConfig(max_cells_per_level=10_000, frontier_min_cells=1), seed=0
        )
        for i in range(6):
            add(a, f"viridian{i}", 8, map_id=M.MAP_IDS["VIRIDIAN_CITY"])
        for i in range(4):
            add(a, f"route1_{i}", 9, map_id=M.MAP_IDS["ROUTE_1"])
        a.set_target_maps({M.MAP_IDS["ROUTE_2"]})
        assert a._target_maps == {M.MAP_IDS["ROUTE_1"]}


class TestDeepest:
    def test_prefers_the_further_map_at_equal_milestone(self):
        """Regression: `play.py --from-frontier` always launched in Oak's Lab.

        Milestone is a worker-lifetime counter, so a worker that reached Route 1 and
        wandered back to Oak's Lab still records milestone 5. Ranking on milestone alone
        made a backtracked cell look like the frontier. `map_rank` is a property of the
        stored state and cannot be inflated that way.
        """
        from pokewm.emulator import maps as M

        a = FrontierArchive(ArchiveConfig(max_cells=16), seed=0)
        add(a, "lab", 5, map_id=M.MAP_IDS["OAKS_LAB"], seen={1, 2, 3, 4, 5})
        add(a, "route", 5, map_id=M.MAP_IDS["ROUTE_1"], seen={1, 2})
        assert a.deepest().key == "route"

    def test_milestone_still_dominates_map_rank(self):
        from pokewm.emulator import maps as M

        a = FrontierArchive(ArchiveConfig(max_cells=16), seed=0)
        add(a, "shallow_far", 2, map_id=M.MAP_IDS["CERULEAN_CITY"])
        add(a, "deep_near", 9, map_id=M.MAP_IDS["PALLET_TOWN"])
        assert a.deepest().key == "deep_near"

    def test_unranked_map_does_not_win(self):
        """A building off the critical path must not outrank a route on it."""
        from pokewm.emulator import maps as M

        a = FrontierArchive(ArchiveConfig(max_cells=16), seed=0)
        add(a, "mart", 5, map_id=M.MAP_IDS["VIRIDIAN_MART"])   # rank -1
        add(a, "route", 5, map_id=M.MAP_IDS["ROUTE_1"])        # rank 5
        assert a.deepest().key == "route"


class TestMapRank:
    def test_follows_the_critical_path_order(self):
        from pokewm.agent.milestones import map_rank
        from pokewm.emulator import maps as M

        order = ["PALLET_TOWN", "OAKS_LAB", "ROUTE_1", "VIRIDIAN_CITY",
                 "ROUTE_2", "PEWTER_CITY", "CERULEAN_CITY", "HALL_OF_FAME"]
        ranks = [map_rank(M.MAP_IDS[n]) for n in order]
        assert ranks == sorted(ranks), ranks
        assert len(set(ranks)) == len(ranks)

    def test_off_path_maps_are_unranked(self):
        from pokewm.agent.milestones import map_rank
        from pokewm.emulator import maps as M

        assert map_rank(M.MAP_IDS["VIRIDIAN_MART"]) == -1

    def test_hall_of_fame_is_last(self):
        from pokewm.agent.milestones import MAP_RANK, map_rank
        from pokewm.emulator import maps as M

        assert map_rank(M.MAP_IDS["HALL_OF_FAME"]) == max(MAP_RANK.values())


class TestStats:
    def test_empty_stats(self, arch):
        s = arch.stats()
        assert s["archive/cells"] == 0.0

    def test_stats_reflect_contents(self, arch):
        add(arch, "a", 2)
        add(arch, "b", 6)
        s = arch.stats()
        assert s["archive/cells"] == 2.0
        assert s["archive/max_milestone"] == 6.0
        assert s["archive/mean_milestone"] == 4.0
        assert 0.0 < s["archive/progress"] <= 1.0


class TestPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        a = FrontierArchive(ArchiveConfig(max_cells=16), seed=0)
        for i in range(5):
            add(a, f"k{i}", milestone=i, seen={i, i + 1}, blob=f"blob{i}".encode(),
                map_id=i, badges=i % 8, events=i * 3)
        a.save(tmp_path / "arch")

        b = FrontierArchive(ArchiveConfig(max_cells=16), seed=0)
        assert b.load(tmp_path / "arch") == 5
        assert len(b) == 5
        assert b.max_milestone == 4
        deepest = b.deepest()
        assert deepest.blob == b"blob4"
        assert deepest.seen_maps == frozenset({4, 5})
        assert deepest.events == 12

    def test_load_missing_directory_is_a_no_op(self, tmp_path):
        a = FrontierArchive(ArchiveConfig(), seed=0)
        assert a.load(tmp_path / "nope") == 0

    def test_repeated_saves_do_not_accumulate_stale_cells(self, tmp_path):
        a = FrontierArchive(ArchiveConfig(max_cells=16), seed=0)
        add(a, "x", 1)
        a.save(tmp_path / "arch")
        add(a, "y", 2)
        a.save(tmp_path / "arch")
        b = FrontierArchive(ArchiveConfig(), seed=0)
        assert b.load(tmp_path / "arch") == 2

    def test_survives_being_saved_while_full(self, tmp_path):
        a = FrontierArchive(ArchiveConfig(max_cells=4), seed=0)
        for i in range(20):
            add(a, f"k{i}", i)
        path = a.save(tmp_path / "arch")
        assert (path / "index.json").exists()
        b = FrontierArchive(ArchiveConfig(max_cells=4), seed=0)
        b.load(path)
        assert b.deepest().milestone == 19


class TestEffectiveTargetIsRecomputed:
    """Regression: the fallback target froze at the instant a milestone fired.

    When the next milestone names an unreached map, the archive falls back to the
    deepest map it currently holds -- a function of the *archive*, which keeps changing,
    not of the milestone. Recomputing it only on milestone transitions sampled it at the
    worst possible instant: the milestone fires the step the agent first enters the new
    map, and cells are inserted at episode end, so the new map is reliably absent.

    Measured after milestone 10: the effective target stuck on Viridian City, the five
    Route 2 cells were drawn once or twice each while Viridian accumulated 26 -- and
    `pokewm.diagnose` reported a healthy 99.7% on-target because it recomputes the
    fallback before sampling, so the diagnostic and the trainer described different
    archives.
    """

    def test_fallback_follows_the_archive_not_the_milestone(self):
        from pokewm.emulator import maps as M

        a = FrontierArchive(
            ArchiveConfig(max_cells_per_level=10_000, frontier_min_cells=1), seed=0
        )
        for i in range(6):
            add(a, f"viridian{i}", 10, map_id=M.MAP_IDS["VIRIDIAN_CITY"])

        # Milestone fires here: Route 2 has no cell yet, so the fallback is Viridian.
        a.set_target_maps({M.MAP_IDS["VIRIDIAN_FOREST"]})
        assert a.target_maps == {M.MAP_IDS["VIRIDIAN_CITY"]}

        # The episode ends and the Route 2 cell lands. Nothing about the milestone has
        # changed, so only a periodic refresh can notice.
        add(a, "route2", 10, map_id=M.MAP_IDS["ROUTE_2"])
        a.set_target_maps({M.MAP_IDS["VIRIDIAN_FOREST"]})
        assert a.target_maps == {M.MAP_IDS["ROUTE_2"]}

    def test_target_maps_exposes_the_effective_set_not_the_request(self):
        """Logging only the request hid the fallback entirely."""
        from pokewm.emulator import maps as M

        a = FrontierArchive(ArchiveConfig(frontier_min_cells=1), seed=0)
        add(a, "route2", 10, map_id=M.MAP_IDS["ROUTE_2"])
        a.set_target_maps({M.MAP_IDS["VIRIDIAN_FOREST"]})
        assert a.target_maps == {M.MAP_IDS["ROUTE_2"]}
        assert M.MAP_IDS["VIRIDIAN_FOREST"] not in a.target_maps


class TestBucketGranularityIsTheRatchetStep:
    """Regression: one bucket spanned the whole approach to the objective.

    The bucket edge is the ratchet step. Inside a single bucket the archive cannot
    distinguish "just arrived" from "at the far edge", so a restore can hand back ground
    the agent had already crossed. Route 2 measured 10 tiles wide by 24 tall: six cells
    at edge 8, of which the archive held five -- saturated, not stalled. Bucket row 6
    spans y 48-55, and the ungated Viridian Forest warp sits at (3, 43), so a cell in
    that row could hold a state saved at y=55, right where the agent arrives from
    Viridian, discarding the whole northward walk. Milestone 10 held 1.3M steps there.
    """

    def test_the_approach_row_is_split_into_rungs(self):
        from pokewm.config import ArchiveConfig as AC

        edge = AC().position_bucket
        # The stretch the agent must hold on to: arrival at y=55 up to the warp at y=43.
        rungs = {y // edge for y in range(43, 56)}
        coarse = {y // 8 for y in range(48, 56)}
        assert len(coarse) == 1, "edge 8 collapsed the whole approach into one cell"
        assert len(rungs) >= 3, f"edge {edge} still banks too little partial progress"

    def test_arrival_and_north_edge_are_no_longer_the_same_cell(self):
        """The specific confusion that made restores cost forward progress."""
        from pokewm.config import ArchiveConfig as AC

        edge, arrival, north_edge = AC().position_bucket, 55, 48
        assert arrival // 8 == north_edge // 8
        assert arrival // edge != north_edge // edge

    def test_a_corridor_yields_more_cells_at_the_configured_edge(self):
        from pokewm.config import ArchiveConfig as AC

        edge = AC().position_bucket
        xs, ys = range(2, 12), range(43, 72)
        fine = {(x // edge, y // edge) for x in xs for y in ys}
        coarse = {(x // 8, y // 8) for x in xs for y in ys}
        assert len(fine) > len(coarse)


class TestHealthAwareSelection:
    """Regression: every restore landed the agent nearly dead.

    On the milestone-11 frontier the archived Viridian Forest cells sat at 10-40% party
    HP. Fighting from there risked a wipe, so fleeing was the correct play -- and the
    agent fled, rationally. A scripted "mash A" policy won 60% of battles from those very
    cells and levelled up, while `level_sum` never exceeded 6.67 across 22M steps. The
    archive scored depth, map rank, target and novelty, and nothing about whether the
    agent was alive enough to act.
    """

    def _archive(self, **kw):
        cfg = ArchiveConfig(restore_prob=1.0, frontier_min_cells=1,
                            max_cells_per_level=10_000, **kw)
        return FrontierArchive(cfg, seed=0)

    def _add(self, a, key, hp, milestone=11, map_id=0):
        return a.insert(key=key, blob=key.encode(), milestone=milestone, map_id=map_id,
                        badges=0, events=0, seen_maps=frozenset({map_id}), hp_frac=hp)

    def test_healthy_cells_are_preferred(self):
        import collections

        a = self._archive()
        for i in range(8):
            self._add(a, f"hurt{i}", hp=0.1)
        for i in range(2):
            self._add(a, f"healthy{i}", hp=1.0)
        picks = collections.Counter()
        for _ in range(600):
            c = a.sample()
            if c:
                picks["healthy" if c.hp_frac > 0.5 else "hurt"] += 1
        frac = picks["healthy"] / max(sum(picks.values()), 1)
        assert frac > 0.5, f"only {frac:.0%} of restores were from a healthy state"

    def test_hurt_cells_are_still_reachable(self):
        """Health is a preference, not a filter -- progress can require a hurt state."""
        import collections

        a = self._archive()
        for i in range(8):
            self._add(a, f"hurt{i}", hp=0.1)
        self._add(a, "healthy", hp=1.0)
        picks = collections.Counter()
        for _ in range(600):
            c = a.sample()
            if c:
                picks[c.key] += 1
        assert sum(v for k, v in picks.items() if k.startswith("hurt")) > 0

    def test_health_does_not_override_a_survivable_frontier(self):
        """A healthy but shallower cell must not outrank a frontier that is merely hurt.

        Narrowed deliberately. This originally asserted that health *never* overrides
        depth, which turned out to be wrong: when every cell at the deepest level is
        unsurvivable the agent is restored into a fight it always loses, and Viridian
        Forest sat that way for 24M steps -- 82 cells, all one level-6 Pokemon at 10-40%
        HP, losing 72 of 90 encounters. `frontier_min_hp` now widens in that case; see
        `TestUnsurvivableFrontierWidens`. Above the threshold the old rule still holds.
        """
        a = self._archive()
        self._add(a, "deep_hurt", hp=0.7, milestone=11)      # scratched, not dying
        self._add(a, "shallow_healthy", hp=1.0, milestone=9)
        picks = sum(1 for _ in range(300) if (a.sample() or a).key == "deep_hurt")
        assert picks > 200

    def test_a_healthier_representative_replaces_the_blob(self):
        a = self._archive()
        self._add(a, "c", hp=0.1)
        self._add(a, "c", hp=0.9)
        assert a._cells["c"].hp_frac == pytest.approx(0.9)

    def test_legacy_cells_without_health_do_not_look_perfect(self, tmp_path):
        """Archives written before health was tracked must not all read as full HP."""
        a = self._archive()
        self._add(a, "known", hp=1.0)
        a.save(tmp_path)
        import json

        idx = tmp_path / "index.json"
        meta = json.loads(idx.read_text())
        for e in meta["cells"] if isinstance(meta, dict) and "cells" in meta else meta:
            e.pop("hp_frac", None)
        idx.write_text(json.dumps(meta))
        b = self._archive()
        b.load(tmp_path)
        assert all(c.hp_frac < 0.5 for c in b._cells.values())


class TestWithinMapRatchet:
    """Regression: nothing distinguished cells inside a single map.

    Milestone, map_rank and the target bonus are all constant across a map, so in a maze
    the size of Viridian Forest the archive had no reason to prefer the cells nearest the
    far exit. The run held milestone 11 for 3.9M steps while covering only the southern
    half. Measured in the forest: 85.9 mean visits near the entrance against 25.5 deep
    inside, corr(y, visits) = +0.73 -- rarely reached means hard to get to, which in a
    maze means further in.
    """

    def _archive(self, **kw):
        cfg = ArchiveConfig(restore_prob=1.0, frontier_min_cells=1,
                            max_cells_per_level=10_000, hp_weight=0.0, **kw)
        return FrontierArchive(cfg, seed=0)

    def _add(self, a, key, visits, milestone=11, map_id=0):
        a.insert(key=key, blob=key.encode(), milestone=milestone, map_id=map_id,
                 badges=0, events=0, seen_maps=frozenset({map_id}), hp_frac=1.0)
        a._cells[key].visits = visits

    def test_rarely_reached_cells_are_preferred(self):
        import collections

        a = self._archive()
        for i in range(15):
            self._add(a, f"entrance{i}", visits=86)   # measured near the entrance
        for i in range(28):
            self._add(a, f"deep{i}", visits=25)       # measured deep inside
        picks = collections.Counter()
        for _ in range(800):
            c = a.sample()
            if c:
                picks["deep" if c.key.startswith("deep") else "entrance"] += 1
        frac = picks["deep"] / max(sum(picks.values()), 1)
        assert frac > 0.75, f"only {frac:.0%} of restores went to the inner frontier"

    def test_the_gap_survives_the_softmax(self):
        """The novelty term's 1/sqrt compressed this to a 0.09 score gap -- invisible."""
        from pokewm.emulator.archive import Cell

        def mk(visits):
            return Cell(key="k", blob=b"", milestone=11, map_id=0, badges=0, events=0,
                        seen_maps=frozenset(), visits=visits)

        cfg = ArchiveConfig()
        gap = (mk(25).score(cfg.novelty_weight, visit_weight=cfg.visit_weight)
               - mk(86).score(cfg.novelty_weight, visit_weight=cfg.visit_weight))
        assert gap > 0.4, f"score gap of {gap:.2f} will vanish under the softmax"

    def test_it_cannot_outrank_the_frontier(self):
        a = self._archive()
        self._add(a, "deep_but_shallow_level", visits=1, milestone=9)
        self._add(a, "frontier", visits=200, milestone=11)
        hits = sum(1 for _ in range(300) if (a.sample() or a).key == "frontier")
        assert hits > 200

    def test_it_cannot_outrank_the_target_bonus(self):
        a = self._archive(target_bonus=6.0, map_rank_weight=1.0)
        self._add(a, "off_target", visits=1, map_id=0)
        self._add(a, "on_target", visits=300, map_id=13)
        a.set_target_maps({13})
        hits = sum(1 for _ in range(300) if (a.sample() or a).map_id == 13)
        assert hits > 200


class TestUnsurvivableFrontierWidens:
    """Regression: every restore landed in a fight the agent could not win.

    `hp_weight` ranks cells within the frontier set, but `frontier_prob` picks that set
    first -- so when the entire deepest level is nearly dead the preference has nothing
    to choose between. Measured: all 82 Viridian Forest cells held one level-6 Pokemon at
    10-40% HP, and an exhaustive search from them lost 72 of 90 wild encounters. The
    party could never level out of the hole it was restored into.
    """

    def _archive(self, **kw):
        # frontier_prob=1.0 isolates the widening rule; at the default 0.8 a fifth of
        # draws are deliberately unrestricted and would mask what is being tested.
        cfg = ArchiveConfig(restore_prob=1.0, frontier_prob=1.0, frontier_min_cells=1,
                            max_cells_per_level=10_000, frontier_min_hp=0.6, **kw)
        return FrontierArchive(cfg, seed=0)

    def _add(self, a, key, hp, milestone):
        a.insert(key=key, blob=key.encode(), milestone=milestone, map_id=0,
                 badges=0, events=0, seen_maps=frozenset({0}), hp_frac=hp)

    def test_a_dying_frontier_falls_back_to_a_healthy_level(self):
        a = self._archive()
        for i in range(6):
            self._add(a, f"dying{i}", hp=0.10, milestone=11)
        for i in range(6):
            self._add(a, f"healthy{i}", hp=1.0, milestone=10)
        drawn = {(a.sample() or a).key for _ in range(300)}
        assert any(k.startswith("healthy") for k in drawn), (
            "the frontier is unsurvivable and no healthy state was offered"
        )

    def test_a_healthy_frontier_is_not_widened(self):
        """Falling back costs progress, so it must only happen when needed."""
        a = self._archive()
        for i in range(6):
            self._add(a, f"fine{i}", hp=1.0, milestone=11)
        for i in range(6):
            self._add(a, f"older{i}", hp=1.0, milestone=10)
        drawn = [(a.sample() or a).milestone for _ in range(300)]
        assert all(m == 11 for m in drawn)

    def test_a_scratched_frontier_is_not_widened(self):
        """0.6 rather than 1.0: mild damage must not trigger a retreat."""
        a = self._archive()
        for i in range(6):
            self._add(a, f"scratched{i}", hp=0.8, milestone=11)
        for i in range(6):
            self._add(a, f"older{i}", hp=1.0, milestone=10)
        drawn = [(a.sample() or a).milestone for _ in range(300)]
        assert all(m == 11 for m in drawn)

    def test_disabled_by_zero(self):
        cfg = ArchiveConfig(restore_prob=1.0, frontier_prob=1.0, frontier_min_cells=1,
                            max_cells_per_level=10_000, frontier_min_hp=0.0)
        a = FrontierArchive(cfg, seed=0)
        for i in range(6):
            self._add(a, f"dying{i}", hp=0.05, milestone=11)
        self._add(a, "healthy", hp=1.0, milestone=10)
        drawn = [(a.sample() or a).milestone for _ in range(200)]
        assert all(m == 11 for m in drawn)


class TestStrongerPartyReplacesTheStoredState:
    """Levels are not part of `progress_key`, so training must survive a restore.

    Measured at 54M env steps: `level_sum` flat at ~5 for the whole run and every one
    of 205 archived Viridian Forest cells holding a single level-6 Pokemon, because the
    `better` test had no level term and a trained-up state was silently discarded.
    """

    def _insert(self, arch, blob, level_sum, hp_frac=0.5):
        return arch.insert(
            key="k", blob=blob, milestone=3, map_id=51, badges=0, events=0,
            seen_maps=frozenset({51}), hp_frac=hp_frac, level_sum=level_sum,
        )

    def test_a_higher_level_state_replaces_the_blob(self):
        arch = FrontierArchive(ArchiveConfig())
        self._insert(arch, b"weak", level_sum=6)
        self._insert(arch, b"strong", level_sum=15)
        assert arch._cells["k"].blob == b"strong"
        assert arch._cells["k"].level_sum == 15

    def test_a_weaker_state_does_not_replace_the_blob(self):
        arch = FrontierArchive(ArchiveConfig())
        self._insert(arch, b"strong", level_sum=15)
        self._insert(arch, b"weak", level_sum=6)
        assert arch._cells["k"].blob == b"strong"
        assert arch._cells["k"].level_sum == 15

    def test_level_survives_a_save_load_round_trip(self, tmp_path):
        arch = FrontierArchive(ArchiveConfig())
        self._insert(arch, b"strong", level_sum=15)
        arch.save(tmp_path / "arch")
        other = FrontierArchive(ArchiveConfig())
        other.load(tmp_path / "arch")
        assert other._cells["k"].level_sum == 15

    def test_an_archive_written_before_levels_were_tracked_loads_as_zero(self, tmp_path):
        """So any state whose strength is known beats one whose strength is not."""
        arch = FrontierArchive(ArchiveConfig())
        self._insert(arch, b"old", level_sum=9)
        arch.save(tmp_path / "arch")
        index = tmp_path / "arch" / "index.json"
        meta = json.loads(index.read_text())
        for entry in meta["cells"]:
            entry.pop("level_sum", None)
        index.write_text(json.dumps(meta))
        other = FrontierArchive(ArchiveConfig())
        other.load(tmp_path / "arch")
        assert other._cells["k"].level_sum == 0


class TestDeepestIsAViableLaunchPad:
    """A cell you cannot survive in is not the most advanced state.

    Measured at 58.7M env steps: `deepest()` returned a Pewter City cell at 0.21 HP
    while 29 of the 62 cells at the same milestone were at full health, and
    `play --from-frontier` blacked out of it in about 100 steps.
    """

    def _add(self, arch, key, hp, milestone=5, map_id=2, seen=None, events=0):
        arch.insert(key=key, blob=key.encode(), milestone=milestone, map_id=map_id,
                    badges=0, events=events,
                    seen_maps=frozenset(seen or {map_id}), hp_frac=hp)

    def test_a_healthy_cell_beats_a_hurt_one_at_the_same_depth(self):
        arch = FrontierArchive(ArchiveConfig())
        self._add(arch, "hurt", hp=0.21)
        self._add(arch, "healthy", hp=1.0)
        assert arch.deepest().key == "healthy"

    def test_depth_still_outranks_health(self):
        """Health is a tie-break, not the objective: a deeper hurt cell still wins."""
        arch = FrontierArchive(ArchiveConfig())
        self._add(arch, "shallow_healthy", hp=1.0, milestone=3)
        self._add(arch, "deep_hurt", hp=0.05, milestone=9)
        assert arch.deepest().key == "deep_hurt"

    def test_all_hurt_still_returns_something(self):
        """The archive must never refuse to name a deepest cell."""
        arch = FrontierArchive(ArchiveConfig())
        self._add(arch, "a", hp=0.1)
        self._add(arch, "b", hp=0.2)
        assert arch.deepest() is not None

    def test_exploration_still_breaks_ties_among_viable_cells(self):
        """Health is thresholded so it does not reorder cells that are all fine."""
        arch = FrontierArchive(ArchiveConfig())
        self._add(arch, "more_maps", hp=0.7, seen={2, 3, 4})
        self._add(arch, "fewer_maps", hp=1.0, seen={2})
        assert arch.deepest().key == "more_maps"


class TestViabilityGate:
    """Health has to gate selection, not merely nudge it.

    Sampled from the live archive at 79.8M env steps: 75% of restores began below 0.3 HP
    and 55% began essentially dead, 47% of them in Pewter City where not one of 24 cells
    was above 0.33 HP. Over the 20.8M steps that followed, `level_sum` never left 8 and
    the heal, ball and party-member rewards never fired once.

    It is a filter rather than a score penalty because selection weight depends on cell
    *count* as much as score: at a penalty of 4.0 the 20 healthy Route 1 cells outweighed
    the 2 healthy North Gate cells and took 66% of restores, seven map-ranks behind the
    frontier; at 2.0 the 24 dying Pewter cells still won.
    """

    def _cell(self, key, hp, level_sum, milestone=13, map_id=2):
        return Cell(key=key, blob=b"x", milestone=milestone, map_id=map_id, badges=0,
                    events=0, seen_maps=frozenset({map_id}), hp_frac=hp,
                    level_sum=level_sum)

    def _arch(self, **kw):
        return FrontierArchive(ArchiveConfig(frontier_min_hp=0.6, frontier_min_cells=6,
                                             **kw), seed=0)

    def test_a_dying_cell_is_not_viable(self):
        a = self._arch()
        assert a.viable(self._cell("d", hp=0.04, level_sum=8)) is False

    def test_a_healthy_cell_is_viable(self):
        a = self._arch()
        assert a.viable(self._cell("h", hp=0.9, level_sum=8)) is True

    def test_an_empty_party_is_viable(self):
        """Pre-starter cells read 0.0 HP because there is nothing to heal.
        `level_sum == 0` identifies them exactly (76/76 on the live archive)."""
        a = self._arch()
        assert a.viable(self._cell("pre", hp=0.0, level_sum=0)) is True

    def test_dying_cells_are_filtered_out_of_the_candidate_set(self):
        a = self._arch()
        cells = ([self._cell(f"h{i}", hp=1.0, level_sum=8) for i in range(6)]
                 + [self._cell(f"d{i}", hp=0.05, level_sum=8) for i in range(20)])
        kept = a._viable_cells(cells)
        assert len(kept) == 6 and all(a.viable(c) for c in kept)

    def test_it_falls_back_when_too_few_are_viable(self):
        """An archive that refuses to restore anything is worse than a hurt restore."""
        a = self._arch()
        cells = ([self._cell("h", hp=1.0, level_sum=8)]
                 + [self._cell(f"d{i}", hp=0.05, level_sum=8) for i in range(20)])
        assert len(a._viable_cells(cells)) == 21

    def test_ordering_among_viable_cells_is_untouched(self):
        """The filter must not reshuffle cells that are all survivable."""
        a = self._arch()
        cells = [self._cell("a", hp=0.7, level_sum=8),
                 self._cell("b", hp=1.0, level_sum=8)]
        assert a._viable_cells(cells) == cells

    def test_disabled_when_not_required(self):
        a = self._arch(require_viable=False)
        cells = [self._cell(f"d{i}", hp=0.05, level_sum=8) for i in range(20)]
        assert len(a._viable_cells(cells)) == 20

    def test_sample_avoids_dying_cells(self):
        a = self._arch(restore_prob=1.0, frontier_prob=0.0)
        for i in range(8):
            a.insert(key=f"h{i}", blob=b"h", milestone=13, map_id=2, badges=0, events=0,
                     seen_maps=frozenset({2}), hp_frac=1.0, level_sum=8)
        for i in range(40):
            a.insert(key=f"d{i}", blob=b"d", milestone=13, map_id=2, badges=0, events=0,
                     seen_maps=frozenset({2}), hp_frac=0.05, level_sum=8)
        picked = [a.sample() for _ in range(200)]
        assert all(c.hp_frac >= 0.6 for c in picked if c is not None)


class TestExperienceRatchets:
    """Levels only move in whole steps; experience is what accumulates.

    A level costs several wins (level 8 -> 9 is ~93 XP for a medium-slow species, a wild
    win pays 20-30), so an archive that ratchets only on `level_sum` discards every
    partial gain and levelling never completes. Measured at 82M env steps: all six
    sampled cells held exactly 327 XP, so not one point had ever been banked, and
    `reward/level` had never fired -- even after `battle_won` got the agent winning.
    """

    def _insert(self, arch, blob, exp, level_sum=8):
        return arch.insert(key="k", blob=blob, milestone=5, map_id=2, badges=0,
                           events=0, seen_maps=frozenset({2}), hp_frac=0.5,
                           level_sum=level_sum, exp=exp)

    def test_more_experience_replaces_the_blob_within_a_level(self):
        arch = FrontierArchive(ArchiveConfig())
        self._insert(arch, b"start", exp=327)
        self._insert(arch, b"progressed", exp=380)      # same level, closer to the next
        assert arch._cells["k"].blob == b"progressed"
        assert arch._cells["k"].exp == 380

    def test_less_experience_does_not_replace_the_blob(self):
        arch = FrontierArchive(ArchiveConfig())
        self._insert(arch, b"progressed", exp=380)
        self._insert(arch, b"start", exp=327)
        assert arch._cells["k"].blob == b"progressed"

    def test_experience_survives_a_save_load_round_trip(self, tmp_path):
        arch = FrontierArchive(ArchiveConfig())
        self._insert(arch, b"x", exp=380)
        arch.save(tmp_path / "a")
        other = FrontierArchive(ArchiveConfig())
        other.load(tmp_path / "a")
        assert other._cells["k"].exp == 380

    def test_an_archive_written_before_exp_was_tracked_loads_as_zero(self, tmp_path):
        arch = FrontierArchive(ArchiveConfig())
        self._insert(arch, b"x", exp=380)
        arch.save(tmp_path / "a")
        index = tmp_path / "a" / "index.json"
        meta = json.loads(index.read_text())
        for entry in meta["cells"]:
            entry.pop("exp", None)
        index.write_text(json.dumps(meta))
        other = FrontierArchive(ArchiveConfig())
        other.load(tmp_path / "a")
        assert other._cells["k"].exp == 0


class TestUtilityCellsSurviveEviction:
    """Pokemon Centers and Poke Marts are launch pads, and eviction deleted them first.

    Victims are ranked by `map_rank`, and no milestone names a shop or a Center, so every
    one scores -1 and loses to anything on the critical path. Measured at 83.5M env
    steps: all 7 Viridian Mart cells were gone, leaving no archived state inside the only
    building that sells Poke Balls -- while every cell carried 0 balls and enough money
    to buy them.
    """

    def _add(self, arch, key, map_id, milestone=2, hp=1.0):
        return arch.insert(key=key, blob=key.encode(), milestone=milestone,
                           map_id=map_id, badges=0, events=0,
                           seen_maps=frozenset({map_id}), hp_frac=hp, level_sum=8)

    def test_a_mart_cell_survives_a_flood_of_on_path_cells(self):
        a = FrontierArchive(ArchiveConfig(max_cells=10_000, max_cells_per_level=4,
                                          utility_cells_per_map=4), seed=0)
        self._add(a, "mart", M.MAP_IDS["VIRIDIAN_MART"])
        for i in range(60):
            self._add(a, f"route{i}", M.MAP_IDS["ROUTE_1"])
        self._add(a, "deeper", M.MAP_IDS["ROUTE_2"], milestone=9)
        assert "mart" in a._cells

    def test_a_pokecenter_cell_survives_too(self):
        a = FrontierArchive(ArchiveConfig(max_cells=10_000, max_cells_per_level=4,
                                          utility_cells_per_map=4), seed=0)
        self._add(a, "center", M.MAP_IDS["VIRIDIAN_POKECENTER"])
        for i in range(60):
            self._add(a, f"route{i}", M.MAP_IDS["ROUTE_1"])
        self._add(a, "deeper", M.MAP_IDS["ROUTE_2"], milestone=9)
        assert "center" in a._cells

    def test_the_shield_is_bounded_per_map(self):
        """Utility cells must not themselves crowd the archive."""
        a = FrontierArchive(ArchiveConfig(max_cells=10_000, max_cells_per_level=4,
                                          utility_cells_per_map=2), seed=0)
        for i in range(30):
            self._add(a, f"mart{i}", M.MAP_IDS["VIRIDIAN_MART"])
        self._add(a, "deeper", M.MAP_IDS["ROUTE_2"], milestone=9)
        marts = [c for c in a._cells.values()
                 if c.map_id == M.MAP_IDS["VIRIDIAN_MART"]]
        assert len(marts) <= 4, len(marts)

    def test_disabled_when_zero(self):
        a = FrontierArchive(ArchiveConfig(max_cells=10_000, max_cells_per_level=4,
                                          utility_cells_per_map=0), seed=0)
        self._add(a, "mart", M.MAP_IDS["VIRIDIAN_MART"])
        for i in range(60):
            self._add(a, f"route{i}", M.MAP_IDS["ROUTE_1"])
        self._add(a, "deeper", M.MAP_IDS["ROUTE_2"], milestone=9)
        assert "mart" not in a._cells


class TestPokecenterCellsStayViable:
    """A hurt party inside a Pokemon Center is a launch pad, not a dead end.

    The viability filter and the utility-cell shield were cancelling each other: the
    shield kept 9 Center cells alive, and the filter then excluded 7 of them -- every
    Pewter one, all at or below 0.25 HP -- which are exactly the states where the heal
    can be practised, six steps and an A press away.
    """

    def _cell(self, key, hp, map_id):
        return Cell(key=key, blob=b"x", milestone=5, map_id=map_id, badges=0, events=0,
                    seen_maps=frozenset({map_id}), hp_frac=hp, level_sum=8)

    def test_a_hurt_pokecenter_cell_is_viable(self):
        a = FrontierArchive(ArchiveConfig(frontier_min_hp=0.6), seed=0)
        assert a.viable(self._cell("c", 0.08, M.MAP_IDS["PEWTER_POKECENTER"]))

    def test_a_hurt_cell_anywhere_else_is_not(self):
        a = FrontierArchive(ArchiveConfig(frontier_min_hp=0.6), seed=0)
        assert not a.viable(self._cell("r", 0.08, M.MAP_IDS["ROUTE_1"]))

    def test_hurt_center_cells_survive_the_candidate_filter(self):
        a = FrontierArchive(ArchiveConfig(frontier_min_hp=0.6, frontier_min_cells=6),
                            seed=0)
        cells = ([self._cell(f"h{i}", 1.0, M.MAP_IDS["ROUTE_1"]) for i in range(6)]
                 + [self._cell("center", 0.08, M.MAP_IDS["PEWTER_POKECENTER"])])
        assert any(c.key == "center" for c in a._viable_cells(cells))


class TestStrengthSurvivesEviction:
    """The strongest party is usually *behind* the frontier, so `map_rank` deletes it.

    It got strong by grinding somewhere safe, which by definition is not the deepest
    map. Measured an hour after the experience ratchet landed: the trim triggered by
    reaching milestone 14 took the archive's best party from 722 XP (level 10) back to
    327 (level 8) -- every stronger state evicted. Strength is not recoverable by
    revisiting a map, unlike coverage.
    """

    def _add(self, arch, key, map_id, exp, milestone=2):
        return arch.insert(key=key, blob=key.encode(), milestone=milestone,
                           map_id=map_id, badges=0, events=0,
                           seen_maps=frozenset({map_id}), hp_frac=1.0,
                           level_sum=8, exp=exp)

    def test_the_strongest_cell_is_not_evicted(self):
        a = FrontierArchive(ArchiveConfig(max_cells=10_000, max_cells_per_level=4,
                                          strongest_cells_kept=2), seed=0)
        self._add(a, "strong", M.MAP_IDS["ROUTE_1"], exp=722)
        for i in range(60):
            self._add(a, f"weak{i}", M.MAP_IDS["ROUTE_2"], exp=327)
        self._add(a, "deeper", M.MAP_IDS["VIRIDIAN_FOREST"], exp=327, milestone=9)
        assert "strong" in a._cells
        assert max(c.exp for c in a._cells.values()) == 722

    def test_a_shallow_strong_cell_beats_a_deep_weak_one(self):
        """The exact ordering that lost the levelling progress."""
        a = FrontierArchive(ArchiveConfig(max_cells=10_000, max_cells_per_level=1,
                                          strongest_cells_kept=1), seed=0)
        self._add(a, "strong_shallow", M.MAP_IDS["ROUTE_1"], exp=722)
        self._add(a, "weak_deep", M.MAP_IDS["ROUTE_2"], exp=100)
        self._add(a, "deeper", M.MAP_IDS["VIRIDIAN_FOREST"], exp=100, milestone=9)
        assert "strong_shallow" in a._cells

    def test_the_shield_is_bounded(self):
        a = FrontierArchive(ArchiveConfig(max_cells=10_000, max_cells_per_level=4,
                                          strongest_cells_kept=2), seed=0)
        for i in range(40):
            self._add(a, f"c{i}", M.MAP_IDS["ROUTE_1"], exp=700 + i)
        self._add(a, "deeper", M.MAP_IDS["ROUTE_2"], exp=100, milestone=9)
        assert len([c for c in a._cells.values() if c.milestone == 2]) <= 6

    def test_zero_experience_cells_are_not_shielded(self):
        """Pre-starter cells have no strength to protect."""
        a = FrontierArchive(ArchiveConfig(max_cells=10_000, max_cells_per_level=2,
                                          strongest_cells_kept=8), seed=0)
        for i in range(30):
            self._add(a, f"pre{i}", M.MAP_IDS["OAKS_LAB"], exp=0)
        self._add(a, "deeper", M.MAP_IDS["ROUTE_2"], exp=100, milestone=9)
        assert len([c for c in a._cells.values() if c.milestone == 2]) <= 2


class TestStrengthReservation:
    """Strength accumulates behind the frontier, so depth-first selection never uses it.

    Sampling the live archive at 87M env steps: 99.5% of restores drew a 327 XP cell and
    0.5% drew anything stronger, with 89% landing in Pewter City. The experience the
    agent had banked sat in post-blackout Pallet Town cells at map_rank 1 and was never
    launched from, so strength could not compound.
    """

    def _add(self, arch, key, map_id, exp, milestone=14, hp=1.0):
        return arch.insert(key=key, blob=key.encode(), milestone=milestone,
                           map_id=map_id, badges=0, events=0,
                           seen_maps=frozenset({map_id}), hp_frac=hp,
                           level_sum=8, exp=exp)

    def _stocked(self, **kw):
        a = FrontierArchive(ArchiveConfig(restore_prob=1.0, **kw), seed=0)
        self._add(a, "strong", M.MAP_IDS["PALLET_TOWN"], exp=400)
        for i in range(40):
            self._add(a, f"deep{i}", M.MAP_IDS["PEWTER_CITY"], exp=327)
        return a

    def test_the_strongest_cell_gets_a_share_of_restores(self):
        """With a pool of 4 the expected share is strength_prob/4 = 3.75%, against a
        measured 0.5% for *everything* stronger than 327 XP before the change."""
        a = self._stocked(strength_prob=0.15, strength_pool=4)
        picks = [a.sample() for _ in range(2000)]
        share = sum(1 for c in picks if c and c.key == "strong") / len(picks)
        assert 0.02 < share < 0.08, share

    def test_a_pool_of_one_sends_the_whole_reservation_to_the_strongest(self):
        a = self._stocked(strength_prob=0.20, strength_pool=1)
        picks = [a.sample() for _ in range(2000)]
        share = sum(1 for c in picks if c and c.key == "strong") / len(picks)
        assert share > 0.15, share

    def test_the_frontier_still_gets_the_majority(self):
        a = self._stocked(strength_prob=0.15, strength_pool=4)
        picks = [a.sample() for _ in range(2000)]
        deep = sum(1 for c in picks if c and c.map_id == M.MAP_IDS["PEWTER_CITY"])
        assert deep / len(picks) > 0.6

    def test_disabled_when_zero(self):
        """Reproduces the measured failure: the strong cell is essentially never drawn."""
        a = self._stocked(strength_prob=0.0)
        picks = [a.sample() for _ in range(2000)]
        share = sum(1 for c in picks if c and c.key == "strong") / len(picks)
        assert share < 0.02, share

    def test_a_dying_strong_cell_is_still_excluded(self):
        """The reservation must not smuggle past the viability filter."""
        a = FrontierArchive(ArchiveConfig(restore_prob=1.0, strength_prob=1.0,
                                          strength_pool=4, frontier_min_hp=0.6,
                                          frontier_min_cells=2), seed=0)
        self._add(a, "strong_dying", M.MAP_IDS["ROUTE_1"], exp=900, hp=0.05)
        for i in range(6):
            self._add(a, f"ok{i}", M.MAP_IDS["ROUTE_1"], exp=327, hp=1.0)
        picks = [a.sample() for _ in range(300)]
        assert all(c.key != "strong_dying" for c in picks if c)

    def test_an_archive_with_no_experience_falls_back(self):
        """Before the starter there is no strength to reserve for."""
        a = FrontierArchive(ArchiveConfig(restore_prob=1.0, strength_prob=1.0), seed=0)
        self._add(a, "pre", M.MAP_IDS["OAKS_LAB"], exp=0)
        assert a.sample() is not None
