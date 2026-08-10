"""Milestone chain properties.

The proof in docs/PROOF.md treats the milestone index as a potential function, which
requires three properties that are easy to break by editing the table: predicates must be
monotone along a trajectory, the chain must be totally ordered and dense, and the
terminal milestone must be exactly the Hall of Fame.
"""

from __future__ import annotations

import pytest

from pokewm.agent import milestones as MS
from pokewm.emulator import maps as M
from pokewm.emulator import ram_map as RM
from pokewm.emulator.ram_map import RamReader

from .conftest import FakeMemory, make_party_mon


def state(map_id=0, badges=0, party=0, dex=0, events=0, events_bits=()) -> RM.GameState:
    mem = FakeMemory()
    mem.write(RM.CUR_MAP, map_id)
    mem.write(RM.BADGES, badges)
    mem.write(RM.PARTY_COUNT, party)
    for i in range(party):
        make_party_mon(mem, i, species=1, level=5, hp=20, max_hp=20)
    if dex:
        mem.write(RM.POKEDEX_OWNED, [0xFF] * (dex // 8) + [(1 << (dex % 8)) - 1])
    if events:
        mem.write(RM.EVENT_FLAGS_START, [0xFF] * (events // 8))
    for idx in events_bits:
        addr = RM.EVENT_FLAGS_START + idx // 8
        mem.write(addr, mem[addr] | (1 << (idx % 8)))
    return RamReader(mem).read()


class TestChainStructure:
    def test_terminal_is_hall_of_fame(self):
        assert MS.MILESTONES[-1].key == "hall_of_fame"
        assert MS.TERMINAL_MILESTONE == MS.NUM_MILESTONES - 1

    def test_keys_unique(self):
        keys = [m.key for m in MS.MILESTONES]
        assert len(set(keys)) == len(keys)

    def test_index_map_is_dense_and_ordered(self):
        for i, m in enumerate(MS.MILESTONES):
            assert MS.MILESTONE_INDEX[m.key] == i

    def test_first_milestone_is_trivially_true(self):
        # The chain must start satisfied, otherwise the prefix length is always 0 and
        # the archive cannot rank anything.
        assert MS.MILESTONES[0].satisfied(state(), [])

    def test_expert_step_budget_is_plausible(self):
        # Sanity bound: a human any% run is ~2 hours of real time. At 24 emulator frames
        # per agent step that is on the order of 10^5 agent steps.
        assert 50_000 < MS.TOTAL_EXPERT_STEPS < 500_000

    def test_all_badge_milestones_present(self):
        badge_keys = {f"badge_{i}" for i in range(1, 9)}
        assert badge_keys <= {m.key for m in MS.MILESTONES}


class TestPredicates:
    def test_badge_predicates_read_the_right_bit(self):
        for bit in range(8):
            gs = state(badges=1 << bit)
            key = {0: "badge_1", 1: "badge_2", 2: "badge_3", 3: "badge_4",
                   4: "badge_5", 5: "badge_6", 6: "badge_7", 7: "badge_8"}[bit]
            m = MS.MILESTONES[MS.MILESTONE_INDEX[key]]
            assert m.satisfied(gs, [])
            assert not m.satisfied(state(badges=0), [])

    def test_hall_of_fame_predicate(self):
        m = MS.MILESTONES[-1]
        assert m.satisfied(state(), [M.MAP_IDS["HALL_OF_FAME"]])
        assert not m.satisfied(state(), [M.MAP_IDS["CHAMPIONS_ROOM"]])

    def test_got_starter_needs_a_party(self):
        m = MS.MILESTONES[MS.MILESTONE_INDEX["got_starter"]]
        assert not m.satisfied(state(party=0), [])
        assert m.satisfied(state(party=1), [])

    def test_visited_predicate_uses_history_not_current_map(self):
        m = MS.MILESTONES[MS.MILESTONE_INDEX["pewter"]]
        # Standing somewhere else but having been to Pewter still counts.
        assert m.satisfied(state(map_id=M.MAP_IDS["ROUTE_3"]),
                           [M.MAP_IDS["PEWTER_CITY"]])


class TestMonotonicity:
    def test_predicates_are_monotone_in_visited_maps(self):
        """Adding maps to history can only ever satisfy more milestones."""
        gs = state(badges=0xFF, party=6, dex=40, events=100)
        seen: set[int] = set()
        prev = 0
        for map_id in range(M.NUM_MAPS):
            seen.add(map_id)
            count = sum(1 for m in MS.MILESTONES if m.satisfied(gs, seen))
            assert count >= prev
            prev = count

    def test_tracker_index_never_decreases(self):
        tracker = MS.MilestoneTracker()
        seq = [
            state(map_id=M.MAP_IDS["REDS_HOUSE_2F"]),
            state(map_id=M.MAP_IDS["REDS_HOUSE_1F"]),
            state(map_id=M.MAP_IDS["PALLET_TOWN"]),
            state(map_id=M.MAP_IDS["OAKS_LAB"], party=1, dex=1),
            state(map_id=M.MAP_IDS["ROUTE_1"], party=1, dex=1),
            state(map_id=M.MAP_IDS["VIRIDIAN_CITY"], party=1, dex=1),
            state(map_id=M.MAP_IDS["PALLET_TOWN"], party=1, dex=1),  # backtracking
        ]
        indices = []
        for gs in seq:
            tracker.update(gs)
            indices.append(tracker.index)
        assert indices == sorted(indices)
        assert indices[-1] >= 6

    def test_tracker_reports_newly_satisfied_once(self):
        tracker = MS.MilestoneTracker()
        first = tracker.update(state(map_id=M.MAP_IDS["PALLET_TOWN"]))
        again = tracker.update(state(map_id=M.MAP_IDS["PALLET_TOWN"]))
        assert "leave_house" in first
        assert again == []

    def test_tracker_roundtrips_through_state_dict(self):
        tracker = MS.MilestoneTracker()
        tracker.update(state(map_id=M.MAP_IDS["VIRIDIAN_CITY"], party=1, dex=1))
        restored = MS.MilestoneTracker()
        restored.load_state_dict(tracker.state_dict())
        assert restored.index == tracker.index
        assert restored.satisfied == tracker.satisfied
        assert restored.seen_maps == tracker.seen_maps

    def test_completed_flag(self):
        tracker = MS.MilestoneTracker()
        assert not tracker.completed
        tracker.update(state(map_id=M.MAP_IDS["HALL_OF_FAME"]))
        assert tracker.completed


class TestAchievedVsNext:
    """Regression: every progress report was off by one.

    `index` is a *count* of satisfied milestones, so `MILESTONES[index]` is the next
    target, not the last achievement. The run logs indexed with the count and announced
    "Reached Viridian City" while the agent had only finished Route 1 and had never
    entered Viridian at all.
    """

    def test_achieved_is_the_previous_entry(self):
        for i in range(1, MS.NUM_MILESTONES + 1):
            assert MS.achieved_milestone(i) is MS.MILESTONES[i - 1]

    def test_nothing_achieved_at_index_zero(self):
        assert MS.achieved_milestone(0) is None

    def test_next_is_the_entry_at_the_count(self):
        for i in range(MS.NUM_MILESTONES):
            assert MS.next_milestone(i) is MS.MILESTONES[i]

    def test_next_is_none_when_complete(self):
        assert MS.next_milestone(MS.NUM_MILESTONES) is None

    def test_achieved_and_next_never_coincide(self):
        for i in range(1, MS.NUM_MILESTONES):
            assert MS.achieved_milestone(i) is not MS.next_milestone(i)

    def test_tracker_labels_match_reality(self):
        """A tracker that has only reached Route 1 must not claim Viridian."""
        tracker = MS.MilestoneTracker()
        for gs in [
            state(map_id=M.MAP_IDS["PALLET_TOWN"]),
            state(map_id=M.MAP_IDS["OAKS_LAB"], party=1, dex=1),
            state(map_id=M.MAP_IDS["ROUTE_1"], party=1, dex=1),
        ]:
            tracker.update(gs)
        assert "Route 1" in tracker.achieved_label
        assert "Viridian" in tracker.frontier_label
        assert tracker.achieved_label != tracker.frontier_label


class TestParcelGate:
    """The parcel run gates the entire rest of the game.

    Viridian's north exit is blocked by an old man until Oak's Parcel is delivered, so
    these two milestones are the difference between progress and a permanent stall. They
    were previously positional proxies: `_visited(a, b)` is ANY-of, so the predicate
    collapsed to "owns a Pokemon" and fired on merely arriving in Viridian -- hiding the
    fact that the agent had never even picked the parcel up in 9.1M steps. Both now key
    on the real story flag.
    """

    def _ms(self, key):
        return MS.MILESTONES[MS.MILESTONE_INDEX[key]]

    def test_visiting_the_maps_does_not_satisfy_either(self):
        everywhere = {M.MAP_IDS[n] for n in
                      ["OAKS_LAB", "VIRIDIAN_CITY", "VIRIDIAN_MART", "PALLET_TOWN"]}
        gs = state(party=1, dex=1)
        assert not self._ms("got_parcel").satisfied(gs, everywhere)
        assert not self._ms("parcel_returned").satisfied(gs, everywhere)

    def test_pickup_keys_on_its_own_flag(self):
        m = self._ms("got_parcel")
        assert m.satisfied(state(events_bits=[RM.EVENT_GOT_OAKS_PARCEL]), set())
        assert not m.satisfied(state(events_bits=[RM.EVENT_GOT_POKEDEX]), set())

    def test_delivery_keys_on_the_measured_delivery_flag(self):
        """EVENT_OAK_GOT_PARCEL, not EVENT_GOT_POKEDEX.

        The delivery flag was watched firing at the exact step the parcel left the bag.
        GOT_POKEDEX's index had been wrong twice, so nothing load-bearing depends on it.
        """
        m = self._ms("parcel_returned")
        assert m.satisfied(state(events_bits=[RM.EVENT_OAK_GOT_PARCEL]), set())
        assert not m.satisfied(state(events_bits=[RM.EVENT_GOT_OAKS_PARCEL]), set())

    def test_pickup_comes_before_delivery(self):
        assert (MS.MILESTONE_INDEX["got_parcel"]
                < MS.MILESTONE_INDEX["parcel_returned"]
                < MS.MILESTONE_INDEX["route_2"])

    def test_targets_send_the_archive_where_the_work_is(self):
        """Each target is the room the interaction happens in, not the route to it.

        Targeting "the way there" (Route 1, Pallet Town) sent 88.8% of restores to
        Route 1 and left the agent rarely starting in Oak's Lab, where a random policy
        completes the delivery in ~830 steps.
        """
        assert self._ms("got_parcel").targets() == {M.MAP_IDS["VIRIDIAN_MART"]}
        assert self._ms("parcel_returned").targets() == {M.MAP_IDS["OAKS_LAB"]}
        # The delivery target must point *backwards* -- not at Viridian, where the
        # forward-only archive bias stalled the run for 9.1M steps.
        assert M.MAP_IDS["VIRIDIAN_CITY"] not in self._ms("parcel_returned").targets()


class TestStatelessIndex:
    def test_matches_tracker_prefix(self):
        seen = {M.MAP_IDS[n] for n in
                ["REDS_HOUSE_1F", "PALLET_TOWN", "OAKS_LAB", "ROUTE_1", "VIRIDIAN_CITY"]}
        gs = state(map_id=M.MAP_IDS["VIRIDIAN_CITY"], party=1, dex=1)
        tracker = MS.MilestoneTracker()
        tracker.seen_maps = set(seen)
        tracker.update(gs)
        assert MS.milestone_index_of(gs, seen) == tracker.index

    def test_zero_when_nothing_reached(self):
        # `boot` is always true, so an empty history still scores 1.
        assert MS.milestone_index_of(state(map_id=0x26), []) >= 1

    @pytest.mark.parametrize("bad", [-1, 999])
    def test_frontier_label_is_clamped(self, bad):
        tracker = MS.MilestoneTracker()
        tracker.index = bad if bad > 0 else 0
        assert isinstance(tracker.frontier_label, str)


class TestEveryGymIsItsOwnObjective:
    """Regression: gyms were rationally ignorable, and never targeted.

    Each gym collapsed to a single "win the badge" milestone. Entering paid only
    `new_map` and a few tiles, walking past cost nothing, and engaging cost `faint` plus
    the HP potential for a fight lost the first several times. A badge also names no map,
    so `targets()` fell back to "the deepest map reached" and the archive never aimed a
    restore at a gym door -- Viridian Gym being the clearest case, since the agent
    crosses Viridian City constantly and never once went in.
    """

    GYMS = {
        "badge_1": ("PEWTER_GYM", 0),
        "badge_2": ("CERULEAN_GYM", 1),
        "badge_3": ("VERMILION_GYM", 2),
        "badge_4": ("CELADON_GYM", 3),
        "badge_5": ("FUCHSIA_GYM", 4),
        "badge_6": ("SAFFRON_GYM", 5),
        "badge_7": ("CINNABAR_GYM", 6),
        "badge_8": ("VIRIDIAN_GYM", 7),
    }

    def _m(self, key):
        return MS.MILESTONES[MS.MILESTONE_INDEX[key]]

    @pytest.mark.parametrize("badge", sorted(GYMS))
    def test_every_gym_has_entry_and_engagement_rungs(self, badge):
        assert f"{badge}_gym" in MS.MILESTONE_INDEX
        assert f"{badge}_engaged" in MS.MILESTONE_INDEX

    @pytest.mark.parametrize("badge", sorted(GYMS))
    def test_rungs_are_ordered_before_the_badge(self, badge):
        assert (MS.MILESTONE_INDEX[f"{badge}_gym"]
                < MS.MILESTONE_INDEX[f"{badge}_engaged"]
                < MS.MILESTONE_INDEX[badge])

    @pytest.mark.parametrize("badge", sorted(GYMS))
    def test_every_badge_points_the_archive_at_its_gym(self, badge):
        gym_map, _ = self.GYMS[badge]
        assert self._m(badge).targets() == {M.MAP_IDS[gym_map]}

    @pytest.mark.parametrize("badge", sorted(GYMS))
    def test_engagement_requires_a_trainer_battle_in_that_gym(self, badge):
        gym_map, _ = self.GYMS[badge]
        gym = M.MAP_IDS[gym_map]

        class GS:
            def __init__(self, map_id, in_battle):
                self.map_id, self.in_battle = map_id, in_battle

        m = self._m(f"{badge}_engaged")
        assert m.satisfied(GS(gym, 2), {gym})
        assert not m.satisfied(GS(gym, 1), {gym}), "a wild battle is not a gym fight"
        assert not m.satisfied(GS(gym, 0), {gym}), "standing in the gym is not fighting"
        assert not m.satisfied(GS(M.MAP_IDS["PALLET_TOWN"], 2), {gym}), "wrong map"

    @pytest.mark.parametrize("badge", sorted(GYMS))
    def test_the_badge_bit_is_preserved(self, badge):
        """Splitting the rungs must not renumber which badge is which."""
        _, bit = self.GYMS[badge]

        class GS:
            def __init__(self, badges):
                self.badges = badges

        assert self._m(badge).predicate(GS(1 << bit), frozenset())
        assert not self._m(badge).predicate(GS(~(1 << bit) & 0xFF), frozenset())

    def test_all_eight_gyms_are_covered(self):
        gyms = {m.targets() and next(iter(m.targets()))
                for m in MS.MILESTONES if m.key.endswith("_gym")}
        assert len(gyms) == 8, f"only {len(gyms)} gyms have an entry rung"

    def test_splitting_did_not_inflate_the_expert_budget(self):
        """The proof's L depends on this total; splitting a rung must be budget-neutral."""
        assert MS.TOTAL_EXPERT_STEPS == 125_900


class TestConnectorMapsAreRanked:
    """A map on the critical path that no milestone names ranks -1, which makes the
    states furthest along score *lowest* in the archive.

    Measured at 57M env steps: the agent had reached Viridian Forest North Gate and the
    three cells there scored 12.6-12.9 against 17.8-18.1 for ordinary Viridian Forest
    cells, with `chosen` = 0 for all three. Every restore went back into the forest.
    """

    def test_the_forest_north_gate_outranks_the_forest(self):
        forest = MS.map_rank(M.MAP_IDS["VIRIDIAN_FOREST"])
        gate = MS.map_rank(M.MAP_IDS["VIRIDIAN_FOREST_NORTH_GATE"])
        assert gate > forest, (gate, forest)

    def test_pewter_still_outranks_the_gate(self):
        assert (MS.map_rank(M.MAP_IDS["PEWTER_CITY"])
                > MS.map_rank(M.MAP_IDS["VIRIDIAN_FOREST_NORTH_GATE"]))

    def test_the_gate_milestone_sits_between_forest_and_pewter(self):
        keys = [m.key for m in MS.MILESTONES]
        assert (keys.index("viridian_forest")
                < keys.index("forest_north_gate")
                < keys.index("pewter"))

    def test_the_gate_rung_fires_on_seeing_the_gate(self):
        gate = MS.MILESTONES[[m.key for m in MS.MILESTONES].index("forest_north_gate")]
        gs = state(map_id=M.MAP_IDS["VIRIDIAN_FOREST_NORTH_GATE"], party=1, dex=1)
        assert gate.satisfied(gs, {M.MAP_IDS["VIRIDIAN_FOREST_NORTH_GATE"]})
        assert not gate.satisfied(gs, {M.MAP_IDS["VIRIDIAN_FOREST"]})

    def test_route_2_alone_still_ranks_below_the_forest(self):
        """Pins the hazard the rung exists to route around: Route 2 spans both sides of
        the forest, and MAP_RANK keeps the earliest milestone that names a map, so
        emerging north onto it cannot be ranked above the forest by map id alone."""
        assert (MS.map_rank(M.MAP_IDS["ROUTE_2"])
                < MS.map_rank(M.MAP_IDS["VIRIDIAN_FOREST"]))
