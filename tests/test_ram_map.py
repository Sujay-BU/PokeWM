"""RAM decoding.

These tests exist because a wrong RAM offset is the single most dangerous class of bug
in this project: it produces a plausible-looking reward signal that is silently
measuring nothing, and training will happily run for a day against it.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokewm.emulator import maps as M
from pokewm.emulator import ram_map as RM
from pokewm.emulator.ram_map import SYMBOLIC_DIM, SYMBOLIC_FEATURES, RamReader, encode_symbolic

from .conftest import FakeMemory, make_party_mon


def _state_with_event(index: int) -> RM.GameState:
    mem = FakeMemory()
    addr = RM.EVENT_FLAGS_START + index // 8
    mem.write(addr, 1 << (index % 8))
    return RamReader(mem).read()


class TestLayout:
    def test_party_struct_stride_matches_known_slot_addresses(self):
        # Independently-documented addresses for slots 2..6 (datacrystal / pokered).
        known_species = [0xD16B, 0xD197, 0xD1C3, 0xD1EF, 0xD21B, 0xD247]
        derived = [RM.PARTY_MON_1 + i * RM.PARTY_MON_STRIDE for i in range(6)]
        assert derived == known_species

    def test_party_struct_field_offsets(self):
        known_level = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        known_hp = [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
        known_maxhp = [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]
        for i in range(6):
            base = RM.PARTY_MON_1 + i * RM.PARTY_MON_STRIDE
            assert base + RM.OFF_LEVEL == known_level[i]
            assert base + RM.OFF_HP == known_hp[i]
            assert base + RM.OFF_MAX_HP == known_maxhp[i]

    def test_party_struct_is_44_bytes(self):
        assert RM.PARTY_MON_STRIDE == 44
        assert RM.OFF_SPECIAL + 2 == RM.PARTY_MON_STRIDE

    def test_event_flag_region_is_320_bytes(self):
        assert RM.EVENT_FLAGS_BYTES == 320
        assert RM.EVENT_FLAGS_END - RM.EVENT_FLAGS_START + 1 == 320

    def test_event_flags_base_address(self):
        """Pinned by measurement, not citation.

        Entering the Viridian Mart runs `SetEvent EVENT_GOT_OAKS_PARCEL`, and the bit
        that actually flips is 0xD74E bit 1. pokered gives that event index 0x29, i.e.
        byte 5 bit 1, so the base is 0xD74E - 5 = 0xD749. The widely-copied 0xD747 is
        wrong and shifts every event index by +16, which silently disabled every
        story-flag milestone.
        """
        assert RM.EVENT_FLAGS_START == 0xD749
        byte = RM.EVENT_FLAGS_START + RM.EVENT_GOT_OAKS_PARCEL // 8
        assert byte == 0xD74E
        assert RM.EVENT_GOT_OAKS_PARCEL % 8 == 1

    def test_pokedex_regions_are_adjacent_and_sized_for_151_species(self):
        assert RM.POKEDEX_BYTES == 19  # ceil(151/8)
        assert RM.POKEDEX_OWNED + RM.POKEDEX_BYTES == RM.POKEDEX_SEEN
        assert RM.POKEDEX_SEEN + RM.POKEDEX_BYTES == RM.NUM_BAG_ITEMS


class TestPrimitives:
    def test_u16_big_endian(self, memory: FakeMemory):
        memory.write(0xC000, [0x01, 0x2C])
        assert RamReader(memory).u16_be(0xC000) == 300

    @pytest.mark.parametrize(
        "digits,expected",
        [([0x00, 0x00, 0x00], 0), ([0x00, 0x30, 0x00], 3000),
         ([0x00, 0x12, 0x34], 1234), ([0x99, 0x99, 0x99], 999999)],
    )
    def test_money_bcd(self, memory: FakeMemory, digits, expected):
        memory.write(RM.MONEY, digits)
        assert RamReader(memory).money() == expected

    def test_money_rejects_invalid_bcd_rather_than_lying(self, memory: FakeMemory):
        memory.write(RM.MONEY, [0x0A, 0x00, 0x00])  # 'A' is not a decimal digit
        assert RamReader(memory).money() == 0

    def test_popcount(self):
        assert RM._popcount_bytes(bytes([0xFF, 0x00, 0x0F])) == 12


class TestGameState:
    def test_decodes_basic_state(self, basic_state: RM.GameState):
        gs = basic_state
        assert gs.map_id == 0x03
        assert gs.map_name == "Cerulean City"
        assert (gs.x, gs.y) == (20, 30)
        assert gs.party_size == 2
        assert gs.badge_count == 2
        assert gs.badge_list == ["boulder", "cascade"]
        assert gs.money == 1234
        assert gs.dex_owned == 10
        assert gs.dex_seen == 16
        assert gs.event_flag_bits == 32

    def test_party_hp_fractions(self, basic_state: RM.GameState):
        assert basic_state.party[0].hp_frac == pytest.approx(0.5)
        assert basic_state.party[1].hp_frac == pytest.approx(1.0)
        assert basic_state.party_hp_frac == pytest.approx(0.75)
        assert basic_state.party_level_sum == 40

    def test_party_wipe_detection(self, memory: FakeMemory):
        memory.write(RM.PARTY_COUNT, 2)
        make_party_mon(memory, 0, 4, 22, 0, 60)
        make_party_mon(memory, 1, 16, 18, 0, 44)
        assert RamReader(memory).read().party_wiped
        make_party_mon(memory, 1, 16, 18, 1, 44)
        assert not RamReader(memory).read().party_wiped

    def test_empty_party_is_not_a_wipe(self, memory: FakeMemory):
        # At the very start of the game the party is empty; treating that as a wipe
        # would terminate every episode on step 1.
        assert not RamReader(memory).read().party_wiped

    def test_uninitialised_party_slot_is_not_a_wipe(self, memory: FakeMemory):
        """Regression: the starter-receipt loop.

        `wPartyCount` is incremented before the Pokemon's stats are written, so for a
        few frames slot 0 reads back as all zeros. Treating max_hp == 0 as "fainted"
        terminated the episode at the exact moment the agent got its first Pokemon,
        which made every downstream milestone unreachable.
        """
        memory.write(RM.PARTY_COUNT, 1)  # count set, struct still zeroed
        gs = RamReader(memory).read()
        assert gs.party_size == 1
        assert gs.live_party == []
        assert not gs.party_wiped
        assert gs.party_hp_frac == 0.0
        assert gs.party_level_sum == 0

    def test_partially_written_party_ignores_only_the_blank_slot(self, memory: FakeMemory):
        memory.write(RM.PARTY_COUNT, 2)
        make_party_mon(memory, 0, species=4, level=10, hp=20, max_hp=20)
        # slot 1 is mid-write: all zeros
        gs = RamReader(memory).read()
        assert gs.party_size == 2
        assert len(gs.live_party) == 1
        assert not gs.party_wiped
        assert gs.party_hp_frac == pytest.approx(1.0)
        assert gs.party_level_sum == 10

    def test_genuine_wipe_is_still_detected(self, memory: FakeMemory):
        """The fix must not mask a real blackout."""
        memory.write(RM.PARTY_COUNT, 2)
        make_party_mon(memory, 0, species=4, level=10, hp=0, max_hp=20)
        make_party_mon(memory, 1, species=16, level=8, hp=0, max_hp=18)
        assert RamReader(memory).read().party_wiped

    def test_party_count_is_clamped_to_six(self, memory: FakeMemory):
        memory.write(RM.PARTY_COUNT, 200)  # corrupt / mid-write
        assert RamReader(memory).read().party_size == 6

    def test_to_text_is_compact_and_mentions_key_facts(self, basic_state):
        text = basic_state.to_text()
        assert "Cerulean City" in text
        assert "boulder" in text
        assert len(text.splitlines()) <= 8


class TestEventFlags:
    """Individual story-flag bits.

    These gate the whole game: the Viridian north exit stays blocked until Oak's Parcel
    is delivered, so a wrong bit index silently makes progress unmeasurable. Indices are
    from pret/pokered's constants/event_constants.asm and validated against real save
    states in the emulator tests.
    """

    def test_bit_addressing(self):
        # bit N lives at byte N//8, bit N%8
        flags = bytes([0b0000_0001, 0b1000_0000]) + bytes(318)
        assert RM.event_bit(flags, 0) == 1
        assert RM.event_bit(flags, 1) == 0
        assert RM.event_bit(flags, 15) == 1
        assert RM.event_bit(flags, 14) == 0

    def test_out_of_range_is_zero_not_an_error(self):
        assert RM.event_bit(bytes(4), 9999) == 0

    def test_known_constants(self):
        # Verbatim from pret/pokered constants/event_constants.asm. GOT_POKEDEX is the
        # 4th const in the first block (index 3); an earlier version took 0x0F from a
        # prose summary and was wrong.
        assert RM.EVENT_FOLLOWED_OAK_INTO_LAB == 0x00
        assert RM.EVENT_GOT_STARTER == 0x08
        assert RM.EVENT_GOT_POKEDEX == 0x0B
        assert RM.EVENT_OAK_GOT_PARCEL == 0x28
        assert RM.EVENT_GOT_OAKS_PARCEL == 0x29

    def test_milestones_key_on_measured_flags(self):
        """The gate milestones must use flags observed firing in the emulator.

        EVENT_GOT_POKEDEX was wrong twice (0x0F, then 0x03) because both values came
        from partial views of the constants file. The two parcel flags were each watched
        flipping in RAM, so the milestones depend on those instead.
        """
        from pokewm.agent.milestones import MILESTONE_INDEX, MILESTONES

        pickup = MILESTONES[MILESTONE_INDEX["got_parcel"]]
        delivery = MILESTONES[MILESTONE_INDEX["parcel_returned"]]
        gs_pick = _state_with_event(RM.EVENT_GOT_OAKS_PARCEL)
        gs_deliver = _state_with_event(RM.EVENT_OAK_GOT_PARCEL)
        assert pickup.satisfied(gs_pick, set())
        assert delivery.satisfied(gs_deliver, set())
        assert not delivery.satisfied(gs_pick, set())

    def test_has_event_on_gamestate(self, memory: FakeMemory):
        gs = RamReader(memory).read()
        assert not gs.has_event(RM.EVENT_GOT_POKEDEX)
        # set bit 0x0B -> byte 1, bit 3
        memory.write(RM.EVENT_FLAGS_START + 1, 0b0000_1000)
        gs = RamReader(memory).read()
        assert gs.has_event(RM.EVENT_GOT_POKEDEX)
        assert not gs.has_event(RM.EVENT_OAK_GOT_PARCEL)

    def test_parcel_flags_are_independent(self, memory: FakeMemory):
        # 0x29 -> byte 5, bit 1
        memory.write(RM.EVENT_FLAGS_START + 5, 0b0000_0010)
        gs = RamReader(memory).read()
        assert gs.has_event(RM.EVENT_GOT_OAKS_PARCEL)
        assert not gs.has_event(RM.EVENT_OAK_GOT_PARCEL)  # 0x28 -> byte 5, bit 0


class TestProgressKey:
    def test_ignores_position(self, memory: FakeMemory):
        memory.write(RM.PARTY_COUNT, 1)
        make_party_mon(memory, 0, 4, 10, 20, 20)
        memory.write(RM.X_COORD, 5)
        a = RamReader(memory).read().progress_key()
        memory.write(RM.X_COORD, 40)
        memory.write(RM.Y_COORD, 12)
        assert RamReader(memory).read().progress_key() == a

    def test_changes_with_badges(self, memory: FakeMemory):
        a = RamReader(memory).read().progress_key()
        memory.write(RM.BADGES, 0x01)
        assert RamReader(memory).read().progress_key() != a

    def test_changes_with_event_flags(self, memory: FakeMemory):
        a = RamReader(memory).read().progress_key()
        memory.write(RM.EVENT_FLAGS_START + 7, 0x08)
        assert RamReader(memory).read().progress_key() != a

    def test_is_deterministic(self, basic_state):
        assert basic_state.progress_key() == basic_state.progress_key()


class TestSymbolicEncoding:
    def test_dimension_matches_feature_names(self, basic_state):
        vec = encode_symbolic(basic_state)
        assert vec.shape == (SYMBOLIC_DIM,) == (len(SYMBOLIC_FEATURES),)
        assert vec.dtype == np.float32

    def test_all_features_bounded(self, basic_state):
        vec = encode_symbolic(basic_state)
        assert np.all(np.isfinite(vec))
        assert vec.min() >= 0.0 and vec.max() <= 4.0

    def test_zero_state_is_finite(self, memory: FakeMemory):
        vec = encode_symbolic(RamReader(memory).read())
        assert np.all(np.isfinite(vec))
        assert vec.max() == 0.0

    def test_badge_feature_tracks_badges(self, memory: FakeMemory):
        i = SYMBOLIC_FEATURES.index("badge_count")
        memory.write(RM.BADGES, 0xFF)
        assert encode_symbolic(RamReader(memory).read())[i] == pytest.approx(1.0)

    def test_enemy_hp_frac_safe_when_max_hp_zero(self, memory: FakeMemory):
        i = SYMBOLIC_FEATURES.index("enemy_hp_frac")
        memory.write_u16_be(RM.ENEMY_MON_HP, 30)
        memory.write_u16_be(RM.ENEMY_MON_MAX_HP, 0)
        assert encode_symbolic(RamReader(memory).read())[i] == 0.0


class TestMapTable:
    def test_table_length(self):
        assert M.NUM_MAPS == 248

    def test_known_ids(self):
        assert M.MAP_IDS["PALLET_TOWN"] == 0x00
        assert M.MAP_IDS["REDS_HOUSE_2F"] == 0x26
        assert M.MAP_IDS["OAKS_LAB"] == 0x28
        assert M.MAP_IDS["VIRIDIAN_FOREST"] == 0x33
        assert M.MAP_IDS["PEWTER_GYM"] == 0x36
        assert M.MAP_IDS["MT_MOON_B2F"] == 0x3D
        assert M.MAP_IDS["HALL_OF_FAME"] == 0x76
        assert M.MAP_IDS["AGATHAS_ROOM"] == 0xF7

    def test_names_are_unique_apart_from_nothing(self):
        assert len(set(M.MAP_NAME_TABLE)) == len(M.MAP_NAME_TABLE)

    def test_every_gym_map_exists(self):
        for bit, map_id in M.GYM_MAPS.items():
            assert 0 <= bit < 8
            assert 0 <= map_id < M.NUM_MAPS

    def test_map_name_handles_out_of_range(self):
        assert "0xFF" in M.map_name(0xFF)

    def test_pokecenters_detected(self):
        assert M.MAP_IDS["VIRIDIAN_POKECENTER"] in M.POKECENTERS
        assert M.MAP_IDS["PALLET_TOWN"] not in M.POKECENTERS
