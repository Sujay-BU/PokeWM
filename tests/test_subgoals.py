"""Subgoal vocabulary and its verification predicates.

The safety argument for putting an 8B model in the loop rests on two claims that are
checked here: the vocabulary is closed and densely indexed (so it can be one-hot encoded
and an out-of-range id is impossible), and every satisfaction predicate is either
progress-monotone or neutral (so a bad suggestion cannot pay out for regressing).
"""

from __future__ import annotations

import pytest

from pokewm.emulator import maps as M
from pokewm.emulator import ram_map as RM
from pokewm.emulator.ram_map import RamReader
from pokewm.llm import subgoals as SG

from .conftest import FakeMemory, make_party_mon


def st(**kw) -> RM.GameState:
    mem = FakeMemory()
    mem.write(RM.CUR_MAP, kw.get("map_id", 0))
    mem.write(RM.X_COORD, kw.get("x", 0))
    mem.write(RM.Y_COORD, kw.get("y", 0))
    mem.write(RM.IS_IN_BATTLE, kw.get("in_battle", 0))
    mem.write(RM.BADGES, kw.get("badges", 0))
    mem.write(RM.TEXT_BOX_ID, kw.get("text_box_id", 0))
    mem.write(RM.NUM_BAG_ITEMS, kw.get("bag", 0))
    mem.write(RM.WALK_BIKE_SURF_STATE, kw.get("wbs", 0))
    mem.write(RM.CURRENT_MENU_ITEM, kw.get("menu", 0))
    mem.write(RM.JOY_IGNORE, kw.get("joy", 0))
    n = kw.get("party", 0)
    mem.write(RM.PARTY_COUNT, n)
    hp = kw.get("hp", 20)
    for i in range(n):
        make_party_mon(mem, i, species=1, level=kw.get("level", 5), hp=hp, max_hp=20)
    if kw.get("events"):
        mem.write(RM.EVENT_FLAGS_START, [0xFF] * kw["events"])
    if kw.get("dex"):
        mem.write(RM.POKEDEX_OWNED, [0xFF] * kw["dex"])
    return RamReader(mem).read()


class TestVocabulary:
    def test_ids_are_dense_and_ordered(self):
        assert [s.id for s in SG.SUBGOALS] == list(range(SG.NUM_SUBGOALS))

    def test_names_unique(self):
        names = [s.name for s in SG.SUBGOALS]
        assert len(set(names)) == len(names)

    def test_every_subgoal_has_a_description(self):
        for s in SG.SUBGOALS:
            assert len(s.description) > 10

    def test_lookup_tables_agree(self):
        for s in SG.SUBGOALS:
            assert SG.BY_NAME[s.name] is s
            assert SG.BY_ID[s.id] is s

    def test_vocabulary_prompt_lists_everything(self):
        prompt = SG.vocabulary_prompt()
        for s in SG.SUBGOALS:
            assert s.name in prompt

    def test_default_subgoal_is_in_range(self):
        assert 0 <= SG.DEFAULT_SUBGOAL < SG.NUM_SUBGOALS


class TestParsing:
    @pytest.mark.parametrize("raw", ["HEAL", "heal", " Heal ", "heal-now", "HEAL_NOW"])
    def test_recognises_variants(self, raw):
        assert SG.parse_subgoal(raw) == SG.BY_NAME["HEAL"].id

    @pytest.mark.parametrize("raw", [None, "", "banana", "{}", "1234"])
    def test_unrecognised_falls_back_to_main_quest(self, raw):
        # This is the safety property: a garbled LLM answer becomes "no guidance",
        # never "wrong guidance".
        assert SG.parse_subgoal(raw) == SG.BY_NAME["MAIN_QUEST"].id

    def test_result_is_always_a_valid_index(self):
        for raw in ["", "???", "WIN_BATTLE", "explore the world", None]:
            assert 0 <= SG.parse_subgoal(raw) < SG.NUM_SUBGOALS


class TestSatisfaction:
    def test_heal_fires_only_on_hp_increase(self):
        before, after = st(party=1, hp=5), st(party=1, hp=20)
        assert SG.satisfied(SG.BY_NAME["HEAL"].id, before, after)
        assert not SG.satisfied(SG.BY_NAME["HEAL"].id, after, before)

    def test_win_battle_requires_leaving_battle_alive(self):
        sid = SG.BY_NAME["WIN_BATTLE"].id
        in_battle = st(party=1, hp=10, in_battle=2)
        won = st(party=1, hp=10, in_battle=0)
        wiped = st(party=1, hp=0, in_battle=0)
        assert SG.satisfied(sid, in_battle, won)
        assert not SG.satisfied(sid, in_battle, wiped)

    def test_challenge_gym_requires_a_new_badge(self):
        sid = SG.BY_NAME["CHALLENGE_GYM"].id
        assert SG.satisfied(sid, st(badges=0), st(badges=1))
        assert not SG.satisfied(sid, st(badges=1), st(badges=1))

    def test_catch_pokemon_on_party_growth_or_new_dex_entry(self):
        sid = SG.BY_NAME["CATCH_POKEMON"].id
        assert SG.satisfied(sid, st(party=1), st(party=2))
        assert SG.satisfied(sid, st(dex=0), st(dex=1))

    def test_leave_area_requires_a_map_change(self):
        sid = SG.BY_NAME["LEAVE_AREA"].id
        assert SG.satisfied(sid, st(map_id=0), st(map_id=12))
        assert not SG.satisfied(sid, st(map_id=0, x=1), st(map_id=0, x=2))

    def test_reach_next_city_only_pays_for_real_cities(self):
        sid = SG.BY_NAME["REACH_NEXT_CITY"].id
        assert SG.satisfied(sid, st(map_id=M.MAP_IDS["ROUTE_1"]),
                            st(map_id=M.MAP_IDS["VIRIDIAN_CITY"]))
        assert not SG.satisfied(sid, st(map_id=M.MAP_IDS["ROUTE_1"]),
                                st(map_id=M.MAP_IDS["ROUTE_2"]))

    def test_none_never_pays(self):
        sid = SG.BY_NAME["NONE"].id
        assert not SG.satisfied(sid, st(), st(badges=0xFF, party=6, events=40))

    def test_advance_dialog_is_verified_on_progress_not_on_text_box_id(self):
        """`wTextBoxID` records the last box type drawn, not whether one is open.

        It is nonzero (1, 13, 20, ...) in the plain overworld, so the intuitive
        predicate `before.text_box_id != 0 and after.text_box_id == 0` could never fire
        and this subgoal was silently unrewardable. It now verifies on a story flag
        advancing, or on a script releasing the joypad.
        """
        sid = SG.BY_NAME["ADVANCE_DIALOG"].id
        # The old, broken condition must NOT be what drives it.
        assert not SG.satisfied(sid, st(text_box_id=5), st(text_box_id=0))
        # Real dialogue outcomes do.
        assert SG.satisfied(sid, st(events=0), st(events=1))
        assert SG.satisfied(sid, st(joy=1), st(joy=0))
        assert not SG.satisfied(sid, st(), st())

    def test_unknown_id_is_never_satisfied(self):
        assert not SG.satisfied(9999, st(), st(badges=0xFF))

    def test_predicates_never_raise(self):
        """A predicate that throws would kill an env worker mid-run."""
        weird = [st(), st(party=6, hp=0), st(badges=0xFF, events=40, dex=19)]
        for sid in range(SG.NUM_SUBGOALS):
            for a in weird:
                for b in weird:
                    assert isinstance(SG.satisfied(sid, a, b), bool)


class TestNoRewardForRegression:
    """No subgoal may pay out for a strictly worse state.

    This is the property that lets docs/PROOF.md §5 conclude the LLM cannot move the
    optimum: every payable event coincides with a non-decrease in some progress
    statistic.
    """

    def test_regression_pays_nothing(self):
        good = st(map_id=M.MAP_IDS["CERULEAN_CITY"], badges=3, party=3, hp=20,
                  events=8, dex=2, bag=5)
        worse = st(map_id=M.MAP_IDS["CERULEAN_CITY"], badges=0, party=0, hp=0,
                   events=0, dex=0, bag=0)
        paying = [sid for sid in range(SG.NUM_SUBGOALS)
                  if SG.satisfied(sid, good, worse)]
        # USE_ITEM legitimately fires on a bag-count decrease (you consumed a potion),
        # which is why it is the sole permitted exception.
        assert paying in ([], [SG.BY_NAME["USE_ITEM"].id])
