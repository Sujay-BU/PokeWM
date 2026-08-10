"""Symbolic decoding of Pokemon Red WRAM.

Addresses follow the `pret/pokered` disassembly symbol names (given in comments as
`wSymbolName`). Every offset here is derived from the WRAM layout in that project;
`tests/test_ram_map.py` re-derives the struct strides and region sizes so a typo
cannot pass silently.

The world model never sees raw RAM. This module turns RAM into a small, stable,
semantically meaningful vector (`GameState`) that is used for three things:

1. the symbolic half of the observation fed to the encoder,
2. milestone / reward computation (`pokewm.agent.rewards`),
3. the textual state summary handed to the LLM subgoal proposer.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------------------
# Scalar addresses
# --------------------------------------------------------------------------------------

# Overworld position / map
CUR_MAP = 0xD35E  # wCurMap
Y_COORD = 0xD361  # wYCoord
X_COORD = 0xD362  # wXCoord
MAP_HEIGHT = 0xD368  # wCurrentMapHeight2 (in 2x2 blocks)
MAP_WIDTH = 0xD369  # wCurrentMapWidth2  (in 2x2 blocks)
CUR_MAP_TILESET = 0xD367  # wCurMapTileset

# Party
PARTY_COUNT = 0xD163  # wPartyCount
PARTY_SPECIES = 0xD164  # wPartySpecies, 6 entries + 0xFF terminator
PARTY_MON_1 = 0xD16B  # wPartyMon1
PARTY_MON_STRIDE = 0x2C  # 44 bytes per party_struct

# Offsets inside a party_struct (see `party_struct` macro in pokered/macros/wram.asm)
OFF_SPECIES = 0x00
OFF_HP = 0x01  # 2 bytes, big-endian
OFF_STATUS = 0x04
# Status byte bits (pokered, `StatusAilments`). Sleep is a counter in the low bits, the
# rest are flags. Poison is the one that matters outside battle: it drains HP every few
# steps in the overworld, so an unnoticed poisoning is a slow death for a small party.
STATUS_SLEEP_MASK = 0b0000_0111
STATUS_POISON = 1 << 3
STATUS_BURN = 1 << 4
STATUS_FREEZE = 1 << 5
STATUS_PARALYSIS = 1 << 6
OFF_TYPE1 = 0x05
OFF_TYPE2 = 0x06
OFF_MOVES = 0x08  # 4 bytes
OFF_PP = 0x1D  # 4 bytes
OFF_LEVEL = 0x21
# 3-byte big-endian experience. Levels are the *visible* unit of strength but experience
# is the one that accumulates: level 8 -> 9 needs ~93 XP for a medium-slow species while
# a wild win pays 20-30, so progress toward a level spans several battles.
OFF_EXP = 0x0E
OFF_MAX_HP = 0x22  # 2 bytes, big-endian
OFF_ATTACK = 0x24
OFF_DEFENSE = 0x26
OFF_SPEED = 0x28
OFF_SPECIAL = 0x2A

# Progress
BADGES = 0xD356  # wObtainedBadges (bitfield)
MONEY = 0xD347  # wPlayerMoney, 3 bytes, big-endian BCD
POKEDEX_OWNED = 0xD2F7  # wPokedexOwned, 19 bytes
POKEDEX_SEEN = 0xD30A  # wPokedexSeen, 19 bytes
POKEDEX_BYTES = 19  # ceil(151 / 8)

# Bag
NUM_BAG_ITEMS = 0xD31D  # wNumBagItems
BAG_ITEMS = 0xD31E  # wBagItems, 20 * (id, qty) then 0xFF
# Master, Ultra, Great, Poke Ball. Without one of these in the bag, catching is not a
# thing the agent can do -- measured: every one of 205 archived Viridian Forest cells
# held a bag of {Town Map, Potion} and a party of exactly one level-6 Pokemon, while the
# CATCH subgoal fired on every wild encounter for tens of millions of steps.
BALL_ITEM_IDS = (0x01, 0x02, 0x03, 0x04)

# Battle
IS_IN_BATTLE = 0xD057  # wIsInBattle: 0 none, 1 wild, 2 trainer
BATTLE_TYPE = 0xD05A  # wBattleType
ENEMY_MON = 0xCFE5  # wEnemyMon (battle_struct)
ENEMY_MON_HP = 0xCFE6  # 2 bytes, big-endian
ENEMY_MON_LEVEL = 0xCFF3  # wEnemyMonLevel
ENEMY_MON_MAX_HP = 0xCFF4  # 2 bytes, big-endian
BATTLE_MON = 0xD014  # wBattleMon
BATTLE_MON_HP = 0xD015  # 2 bytes, big-endian
BATTLE_MON_MAX_HP = 0xD023  # 2 bytes, big-endian

# UI / menu state
TOP_MENU_ITEM_Y = 0xCC24  # wTopMenuItemY
TOP_MENU_ITEM_X = 0xCC25  # wTopMenuItemX
CURRENT_MENU_ITEM = 0xCC26  # wCurrentMenuItem
MAX_MENU_ITEM = 0xCC28  # wMaxMenuItem
MENU_WATCHED_KEYS = 0xCC29  # wMenuWatchedKeys
# wListScrollOffset: which entry of a long list sits at the top of the visible window.
# Gen 1 keeps `wCurrentMenuItem` *within* the window, so it saturates at the last
# visible row while the page scrolls underneath. Verified in the Viridian Mart list:
# pressing down gave item=2/scroll=0, then item=2/scroll=1, then item=2/scroll=2 -- three
# different items with identical menu bytes.
LIST_SCROLL_OFFSET = 0xCC36
# wItemQuantity: the count in the buy/sell/toss quantity selector. Verified stepping
# 1,2,3,4,5 on successive UP presses while every menu byte stayed constant.
ITEM_QUANTITY = 0xCF96
TEXT_BOX_ID = 0xD125  # wTextBoxID
JOY_IGNORE = 0xCD6B  # wJoyIgnore -- nonzero while a script owns input

# Movement / world state
WALK_BIKE_SURF_STATE = 0xD700  # wWalkBikeSurfState: 0 walk, 1 bike, 2 surf
D730 = 0xD730  # wd730 (bit 7: script-controlled movement)
D733 = 0xD733  # wd733
PLAYER_MOVING = 0xCFC6  # wPlayerMovingDirection-ish; used only as a heuristic

# Event flag regions. `wEventFlags` is 320 bytes (2560 individual flags) and is the
# canonical record of "what has happened" in a save file.
# wEventFlags. Note 0xD749, not the 0xD747 that circulates in several Pokemon RL
# projects. Established by measurement, not by citation: entering the Viridian Mart
# triggers `SetEvent EVENT_GOT_OAKS_PARCEL`, and the bit that actually flips is at
# 0xD74E bit 1. pokered gives that event index 0x29 -> byte 5, bit 1, so the base must be
# 0xD74E - 5 = 0xD749. The bit-within-byte matching exactly is what pins it.
#
# Using 0xD747 shifted every event index by +16, which silently broke every story-flag
# predicate: `got_parcel` read a bit that is never set, so a milestone the agent was
# achieving within ~25 steps of entering the Mart looked permanently unreachable.
EVENT_FLAGS_START = 0xD749  # wEventFlags
EVENT_FLAGS_BYTES = 320
EVENT_FLAGS_END = EVENT_FLAGS_START + EVENT_FLAGS_BYTES - 1  # inclusive

MISSABLE_OBJECT_FLAGS = 0xD5A6  # wMissableObjectFlags, 32 bytes
MISSABLE_OBJECT_BYTES = 32

# Story-flag bit indices into wEventFlags, taken verbatim from `pret/pokered`'s
# constants/event_constants.asm. Event N lives at byte `EVENT_FLAGS_START + N // 8`,
# bit `N % 8`.
#
# These are only correct because EVENT_FLAGS_START was corrected above.
#
# EVENT_GOT_POKEDEX was wrong twice before this: 0x0F from a prose summary of the
# constants file, then 0x03 from a *grep-filtered* view of the file, which silently
# renumbers everything because the index is the constant's position in the block. It is
# 0x0B. The lesson is that any index derived from a partial view of the source is a
# guess. Prefer constants that have been observed firing in the emulator -- the two
# parcel flags below both were, and the milestones depend on those rather than on
# GOT_POKEDEX for exactly that reason.
EVENT_FOLLOWED_OAK_INTO_LAB = 0x00
EVENT_GOT_TOWN_MAP = 0x03
EVENT_FOLLOWED_OAK_INTO_LAB_2 = 0x06
EVENT_OAK_ASKED_TO_CHOOSE_MON = 0x07
EVENT_GOT_STARTER = 0x08
EVENT_BATTLED_RIVAL_IN_OAKS_LAB = 0x09
EVENT_GOT_POKEBALLS_FROM_OAK = 0x0A
EVENT_GOT_POKEDEX = 0x0B
EVENT_OAK_GOT_PARCEL = 0x28   # measured: fires the step the parcel leaves the bag
EVENT_GOT_OAKS_PARCEL = 0x29  # measured: 0xD74E bit 1 on Viridian Mart entry


def event_bit(flags: bytes, index: int) -> int:
    """Read one story flag out of a wEventFlags snapshot."""
    byte = index // 8
    if byte >= len(flags):
        return 0
    return (flags[byte] >> (index % 8)) & 1

# A single flag that is set by the intro and is *not* part of real progress; excluding it
# keeps the "events completed" count honest at t=0.
EVENT_FLAG_EXCLUSIONS = frozenset({(0xD74B, 5)})  # EVENT_GOT_TOWN_MAP-adjacent noise

BADGE_NAMES = (
    "boulder",  # Brock,     Pewter
    "cascade",  # Misty,     Cerulean
    "thunder",  # Surge,     Vermilion
    "rainbow",  # Erika,     Celadon
    "soul",  # Koga,      Fuchsia
    "marsh",  # Sabrina,   Saffron
    "volcano",  # Blaine,    Cinnabar
    "earth",  # Giovanni,  Viridian
)

# --------------------------------------------------------------------------------------
# Map ids -- the full table lives in `maps.py`; a few are re-exported for convenience.
# --------------------------------------------------------------------------------------

MAP_PALLET_TOWN = 0x00
MAP_VIRIDIAN_CITY = 0x01
MAP_PEWTER_CITY = 0x02
MAP_CERULEAN_CITY = 0x03
MAP_LAVENDER_TOWN = 0x04
MAP_VERMILION_CITY = 0x05
MAP_CELADON_CITY = 0x06
MAP_FUCHSIA_CITY = 0x07
MAP_CINNABAR_ISLAND = 0x08
MAP_INDIGO_PLATEAU = 0x09
MAP_SAFFRON_CITY = 0x0A
MAP_ROUTE_1 = 0x0C
MAP_ROUTE_2 = 0x0D
MAP_ROUTE_3 = 0x0E
MAP_ROUTE_4 = 0x0F
MAP_ROUTE_22 = 0x21
MAP_REDS_HOUSE_1F = 0x25
MAP_REDS_HOUSE_2F = 0x26
MAP_OAKS_LAB = 0x28
MAP_VIRIDIAN_POKECENTER = 0x29
MAP_VIRIDIAN_MART = 0x2A
MAP_PEWTER_GYM = 0x36
MAP_MT_MOON_1F = 0x3B
MAP_MT_MOON_B1F = 0x3C
MAP_MT_MOON_B2F = 0x3D
MAP_CERULEAN_GYM = 0x41
MAP_ROCK_TUNNEL_1F = 0x52
MAP_SS_ANNE_1F = 0x5F
MAP_VERMILION_GYM = 0x5C
MAP_HALL_OF_FAME = 0x76

from .maps import map_name  # noqa: E402  (kept below the constants above)

# --------------------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------------------


def _u16_be(lo_hi: tuple[int, int]) -> int:
    """Game Boy multi-byte stats in the party struct are stored big-endian."""
    return (lo_hi[0] << 8) | lo_hi[1]


def _bcd3(digits: tuple[int, int, int]) -> int:
    """Money is 3 bytes of packed binary-coded decimal, most significant first."""
    total = 0
    for byte in digits:
        hi, lo = byte >> 4, byte & 0x0F
        if hi > 9 or lo > 9:  # corrupt/mid-write value; fall back to 0 rather than lie
            return 0
        total = total * 100 + hi * 10 + lo
    return total


def _popcount_bytes(data: bytes | bytearray | np.ndarray) -> int:
    arr = np.frombuffer(bytes(data), dtype=np.uint8)
    return int(np.unpackbits(arr).sum())


@dataclass(frozen=True)
class PartyMon:
    species: int
    level: int
    hp: int
    max_hp: int
    status: int
    exp: int = 0

    @property
    def alive(self) -> bool:
        return self.hp > 0

    @property
    def hp_frac(self) -> float:
        return self.hp / self.max_hp if self.max_hp > 0 else 0.0


@dataclass
class GameState:
    """A decoded snapshot of everything the agent is allowed to know."""

    map_id: int
    x: int
    y: int
    map_w: int
    map_h: int
    tileset: int

    party: list[PartyMon]
    badges: int  # bitfield
    money: int
    dex_owned: int
    dex_seen: int

    in_battle: int  # 0 none, 1 wild, 2 trainer
    battle_type: int
    enemy_hp: int
    enemy_max_hp: int
    enemy_level: int

    menu_item: int
    max_menu_item: int
    # Screen position of the menu's first entry. In a *grid* menu this is the only
    # place the column lives: the Pokemon Red battle menu is 2x2 with max_menu_item=1,
    # so `menu_item` alone cannot tell FIGHT from ITEM or PKMN from RUN.
    menu_top_y: int
    menu_top_x: int
    list_scroll: int
    item_quantity: int
    text_box_id: int
    joy_ignore: int
    walk_bike_surf: int

    event_flag_bits: int  # popcount over wEventFlags
    missable_bits: int  # popcount over wMissableObjectFlags
    event_flags_raw: bytes = field(repr=False, default=b"")
    bag_item_count: int = 0
    ball_count: int = 0  # total Poke/Great/Ultra/Master Balls carried

    # ---- derived -------------------------------------------------------------------

    @property
    def badge_count(self) -> int:
        return int(bin(self.badges).count("1"))

    @property
    def badge_list(self) -> list[str]:
        return [n for i, n in enumerate(BADGE_NAMES) if self.badges & (1 << i)]

    @property
    def party_size(self) -> int:
        return len(self.party)

    @property
    def live_party(self) -> list[PartyMon]:
        """Party slots whose data has actually been written.

        `wPartyCount` is incremented *before* the Pokemon's stats are copied into the
        party struct, so for a few frames during the starter-receipt script (and every
        capture and every Pokecenter withdrawal) slot 0 reads back as all zeros:
        species 0, level 0, hp 0, max_hp 0.

        Treating `max_hp == 0` as "fainted" made `party_wiped` fire on that transient,
        which terminated the episode at the exact instant the agent received its first
        Pokemon. The agent then restored from the archive, walked back to Oak, received
        the starter again, and terminated again -- an endless loop that made every
        milestone past `got_starter` unreachable and pinned `party_level_sum` at 0 for
        4.6M steps. A slot with `max_hp == 0` is uninitialised, not dead.
        """
        return [m for m in self.party if m.max_hp > 0]

    @property
    def party_level_sum(self) -> int:
        return sum(m.level for m in self.live_party)

    @property
    def party_exp_sum(self) -> int:
        """Total party experience -- the strength measure that actually accumulates.

        `party_level_sum` only moves in whole levels, and a level costs several wins, so
        an archive that ratchets on levels alone discards every partial gain. Measured:
        all six sampled cells held exactly 327 XP after 82M env steps, so not one point
        of experience had ever been banked.
        """
        return sum(m.exp for m in self.party)

    @property
    def party_hp_frac(self) -> float:
        live = self.live_party
        if not live:
            return 0.0
        return float(np.mean([m.hp_frac for m in live]))

    @property
    def party_statused(self) -> float:
        """Fraction of the live party carrying any status ailment.

        Read from RAM since the beginning but never surfaced, so the agent could not see
        that it was poisoned, asleep or paralysed -- the same blind spot the battle menu
        cursor had. Poison is the pressing one: it costs HP every few steps in the
        overworld, so a poisoned party bleeds out while walking and the agent had no
        signal that anything was wrong.
        """
        live = self.live_party
        if not live:
            return 0.0
        return float(np.mean([1.0 if m.status else 0.0 for m in live]))

    @property
    def party_poisoned(self) -> bool:
        """Any live member poisoned -- the ailment that drains HP as the agent walks."""
        return any(m.status & STATUS_POISON for m in self.live_party)

    @property
    def party_wiped(self) -> bool:
        live = self.live_party
        return bool(live) and all(not m.alive for m in live)

    @property
    def position(self) -> tuple[int, int, int]:
        return (self.map_id, self.x, self.y)

    def has_event(self, index: int) -> bool:
        """Whether a specific story flag is set. See EVENT_* constants above."""
        return bool(event_bit(self.event_flags_raw, index))

    @property
    def map_name(self) -> str:
        return map_name(self.map_id)

    def progress_key(self) -> str:
        """Stable hash of *irreversible* progress.

        Used as the archive cell key for the Go-Explore style frontier (see
        `pokewm.emulator.archive`). Deliberately excludes x/y and HP so that wandering
        around a town does not spawn thousands of near-duplicate cells.
        """
        h = hashlib.blake2b(digest_size=8)
        h.update(bytes([self.badges, self.dex_owned & 0xFF, min(self.party_size, 6)]))
        h.update(self.event_flags_raw)
        return h.hexdigest()

    def to_text(self) -> str:
        """Human/LLM-readable summary. Kept short: it is re-sent on every LLM call."""
        party = ", ".join(
            f"#{m.species} Lv{m.level} {m.hp}/{m.max_hp}HP" for m in self.party
        ) or "empty"
        battle = {0: "no", 1: "wild battle", 2: "trainer battle"}.get(self.in_battle, "?")
        return (
            f"location: {self.map_name} at (x={self.x}, y={self.y})\n"
            f"badges: {self.badge_count} [{', '.join(self.badge_list) or 'none'}]\n"
            f"party ({self.party_size}/6): {party}\n"
            f"money: ${self.money}  pokedex: {self.dex_owned} owned / {self.dex_seen} seen\n"
            f"in battle: {battle}\n"
            f"story flags set: {self.event_flag_bits}"
        )


class RamReader:
    """Reads a `GameState` out of any object exposing a `memory`-like mapping.

    PyBoy's `pyboy.memory` supports both scalar and slice indexing. Tests substitute a
    plain bytearray-backed fake, so this class must not touch anything else on PyBoy.
    """

    def __init__(self, memory):
        self._m = memory

    # -- primitives --------------------------------------------------------------

    def byte(self, addr: int) -> int:
        return int(self._m[addr]) & 0xFF

    def block(self, addr: int, length: int) -> bytes:
        return bytes(bytearray(self._m[addr : addr + length]))

    def u16_be(self, addr: int) -> int:
        raw = self.block(addr, 2)
        return _u16_be((raw[0], raw[1]))

    # -- structured --------------------------------------------------------------

    def party(self) -> list[PartyMon]:
        count = min(self.byte(PARTY_COUNT), 6)
        mons: list[PartyMon] = []
        for i in range(count):
            base = PARTY_MON_1 + i * PARTY_MON_STRIDE
            mons.append(
                PartyMon(
                    species=self.byte(base + OFF_SPECIES),
                    level=self.byte(base + OFF_LEVEL),
                    hp=self.u16_be(base + OFF_HP),
                    max_hp=self.u16_be(base + OFF_MAX_HP),
                    status=self.byte(base + OFF_STATUS),
                    exp=(self.byte(base + OFF_EXP) << 16)
                    | (self.byte(base + OFF_EXP + 1) << 8)
                    | self.byte(base + OFF_EXP + 2),
                )
            )
        return mons

    def money(self) -> int:
        raw = self.block(MONEY, 3)
        return _bcd3((raw[0], raw[1], raw[2]))

    def event_flags(self) -> bytes:
        return self.block(EVENT_FLAGS_START, EVENT_FLAGS_BYTES)

    def read(self) -> GameState:
        events = self.event_flags()
        return GameState(
            map_id=self.byte(CUR_MAP),
            x=self.byte(X_COORD),
            y=self.byte(Y_COORD),
            map_w=self.byte(MAP_WIDTH),
            map_h=self.byte(MAP_HEIGHT),
            tileset=self.byte(CUR_MAP_TILESET),
            party=self.party(),
            badges=self.byte(BADGES),
            money=self.money(),
            dex_owned=_popcount_bytes(self.block(POKEDEX_OWNED, POKEDEX_BYTES)),
            dex_seen=_popcount_bytes(self.block(POKEDEX_SEEN, POKEDEX_BYTES)),
            in_battle=self.byte(IS_IN_BATTLE),
            battle_type=self.byte(BATTLE_TYPE),
            enemy_hp=self.u16_be(ENEMY_MON_HP),
            enemy_max_hp=self.u16_be(ENEMY_MON_MAX_HP),
            enemy_level=self.byte(ENEMY_MON_LEVEL),
            menu_item=self.byte(CURRENT_MENU_ITEM),
            max_menu_item=self.byte(MAX_MENU_ITEM),
            menu_top_y=self.byte(TOP_MENU_ITEM_Y),
            menu_top_x=self.byte(TOP_MENU_ITEM_X),
            list_scroll=self.byte(LIST_SCROLL_OFFSET),
            item_quantity=self.byte(ITEM_QUANTITY),
            text_box_id=self.byte(TEXT_BOX_ID),
            joy_ignore=self.byte(JOY_IGNORE),
            walk_bike_surf=self.byte(WALK_BIKE_SURF_STATE),
            event_flag_bits=_popcount_bytes(events),
            missable_bits=_popcount_bytes(
                self.block(MISSABLE_OBJECT_FLAGS, MISSABLE_OBJECT_BYTES)
            ),
            event_flags_raw=events,
            bag_item_count=min(self.byte(NUM_BAG_ITEMS), 20),
            ball_count=self._ball_count(),
        )

    def _ball_count(self) -> int:
        """Total balls in the bag. The list is (id, qty) pairs terminated by 0xFF."""
        n = min(self.byte(NUM_BAG_ITEMS), 20)
        total = 0
        for i in range(n):
            item_id = self.byte(BAG_ITEMS + 2 * i)
            if item_id == 0xFF:
                break
            if item_id in BALL_ITEM_IDS:
                total += self.byte(BAG_ITEMS + 2 * i + 1)
        return total


# --------------------------------------------------------------------------------------
# Symbolic feature vector
# --------------------------------------------------------------------------------------

# Order matters: the encoder's input width is derived from this list, and tests assert
# that SYMBOLIC_FEATURES stays in sync with `encode_symbolic`.
SYMBOLIC_FEATURES = (
    "map_id_norm",
    "x_norm",
    "y_norm",
    "badge_count",
    "party_size",
    "party_level_mean",
    "party_hp_frac",
    "party_statused",
    "party_poisoned",
    "party_wiped",
    "money_log",
    "dex_owned",
    "dex_seen",
    "in_battle_wild",
    "in_battle_trainer",
    "enemy_hp_frac",
    "enemy_level",
    # `menu_active` used to live here as `float(max_menu_item > 0)`. It was a dead
    # input: `wMaxMenuItem` is never cleared when a menu closes, so it reads nonzero
    # in the plain overworld too -- measured 1119 of 1119 archived cells with no menu
    # open and no script running. Exactly the `wTextBoxID` trap one field below.
    #
    # Which entry the cursor sits on, not just whether a menu is open. The cursor is
    # drawn on screen, but at 72x80 downsampled luminance it is a couple of pixels,
    # which is not a signal a conv encoder will reliably pick out.
    "menu_cursor",
    # The cursor's *absolute* index, because `menu_cursor` is a ratio: entry 1 of 2 and
    # entry 3 of 6 both read 0.5, and a stale `max_menu_item` of 1 with `menu_item` 3
    # reads 3.0.
    "menu_index",
    # Where the menu is drawn. This is what makes the battle menu legible at all: it is
    # a 2x2 grid with max_menu_item=1, so the row is in `menu_item` and the column is
    # only in `menu_top_x` (9 = FIGHT/PKMN, 15 = ITEM/RUN). Without the column, FIGHT
    # and ITEM are the same observation, and so are PKMN and RUN -- the agent could not
    # tell attacking from opening the bag. `menu_top_y` likewise separates the battle
    # menu (14) from the move list (12) from a text box (2).
    "menu_row_origin",
    "menu_col_origin",
    # A long list scrolls under a fixed cursor window, so `menu_index` alone identifies
    # a *row on screen*, not an entry. Measured in the Viridian Mart: three different
    # items all read item=2, differing only in scroll. `list_index` is the entry's
    # absolute position, which is what "select the Poke Ball" actually means.
    "list_scroll",
    "list_index",
    # The buy/sell/toss quantity. Nothing else moves when it changes -- the selector was
    # indistinguishable from the list it opens over, and the count invisible.
    "item_quantity",
    "script_active",
    "surfing",
    "biking",
    "event_bits",
    "missable_bits",
    "bag_items",
    # Whether catching is possible at all. `bag_items` counts *distinct* item slots, so
    # a bag holding a Town Map and one holding six Poke Balls look the same to it -- the
    # agent could not tell "throw a ball" from "there is no ball to throw".
    "balls",
    "has_ball",
)
SYMBOLIC_DIM = len(SYMBOLIC_FEATURES)


def encode_symbolic(gs: GameState) -> np.ndarray:
    """Map a GameState to a bounded float32 vector.

    Everything is scaled into roughly [0, 1] so the encoder sees a well-conditioned
    input without needing per-feature normalisation statistics that would drift as the
    agent reaches new parts of the game.
    """
    out = np.array(
        [
            gs.map_id / 255.0,
            gs.x / 64.0,
            gs.y / 64.0,
            gs.badge_count / 8.0,
            gs.party_size / 6.0,
            (gs.party_level_sum / max(gs.party_size, 1)) / 100.0,
            gs.party_hp_frac,
            gs.party_statused,
            float(gs.party_poisoned),
            float(gs.party_wiped),
            np.log1p(gs.money) / np.log1p(999999.0),
            gs.dex_owned / 151.0,
            gs.dex_seen / 151.0,
            float(gs.in_battle == 1),
            float(gs.in_battle == 2),
            (gs.enemy_hp / gs.enemy_max_hp) if gs.enemy_max_hp > 0 else 0.0,
            gs.enemy_level / 100.0,
            (gs.menu_item / gs.max_menu_item) if gs.max_menu_item > 0 else 0.0,
            min(gs.menu_item, 15) / 15.0,
            # Screen is 18 rows x 20 columns of tiles.
            min(gs.menu_top_y, 17) / 17.0,
            min(gs.menu_top_x, 19) / 19.0,
            min(gs.list_scroll, 20) / 20.0,
            min(gs.menu_item + gs.list_scroll, 20) / 20.0,
            min(gs.item_quantity, 99) / 99.0,
            # `wTextBoxID` records the *last* text box type drawn, not whether one is
            # currently on screen: it is nonzero (1, 13, 20, ...) in the plain overworld
            # too. Including it made this feature a constant 1 -- a dead input the
            # encoder could learn nothing from. `wJoyIgnore` genuinely toggles, being
            # nonzero only while a script owns the joypad.
            float(gs.joy_ignore != 0),
            float(gs.walk_bike_surf == 2),
            float(gs.walk_bike_surf == 1),
            gs.event_flag_bits / 320.0,
            gs.missable_bits / 64.0,
            gs.bag_item_count / 20.0,
            min(gs.ball_count, 10) / 10.0,
            float(gs.ball_count > 0),
        ],
        dtype=np.float32,
    )
    return np.clip(out, 0.0, 4.0)
