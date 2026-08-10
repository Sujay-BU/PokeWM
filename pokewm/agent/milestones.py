"""The critical path of Pokemon Red, expressed as a totally ordered milestone chain.

Why this exists
---------------
Pokemon Red is a ~10^5-step-horizon sparse-reward problem. No credit-assignment scheme
survives that horizon directly. Every system that has actually finished the game --
Gemini 2.5 Pro's 2025 run, the Claude Plays Pokemon harness -- did so by decomposing it
into an ordered list of objectives and solving them one at a time.

We do the same, but the decomposition is used for three *mechanical* purposes rather
than as a script:

1. **Archive scoring** (`pokewm.emulator.archive`): the milestone index of a save state
   is its primary score, so the frontier archive always knows which cell is "furthest".
2. **Curriculum**: episodes are launched from cells at or just behind the frontier, so
   the effective horizon per episode is the length of *one* milestone, not the game.
3. **The proof** (docs/PROOF.md): the chain is the sequence of subproblems whose
   per-attempt success probabilities multiply into the completion bound.

Predicates are written against `GameState` only -- observable RAM, no scripted inputs.
They are deliberately *monotone*: once true for a trajectory they stay true, which is
what makes the milestone index a valid potential function.

Milestone predicates are checked against a running maximum of visited maps and the
current state, so "reached Pewter City" stays satisfied after you leave Pewter City.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from ..emulator import maps as M
from ..emulator import ram_map as RM
from ..emulator.ram_map import GameState

# A predicate sees the current state plus the set of map ids visited so far this run.
Predicate = Callable[[GameState, "frozenset[int]"], bool]


def _map_ids_of(pred: Predicate) -> frozenset[int]:
    """Map ids a predicate keys on, if any. Used to build the map-rank table."""
    return getattr(pred, "map_ids", frozenset())


def _visited(*names: str) -> Predicate:
    ids = frozenset(M.MAP_IDS[n] for n in names)

    def pred(gs, seen):
        return bool(ids & seen)

    pred.map_ids = ids  # type: ignore[attr-defined]
    return pred


def _badge(bit: int) -> Predicate:
    return lambda gs, seen: bool(gs.badges & (1 << bit))


def _party(n: int) -> Predicate:
    return lambda gs, seen: gs.party_size >= n


def _fighting_in(*names: str) -> Predicate:
    """In a *trainer* battle while standing on one of these maps.

    A gym leader is a trainer battle on the gym map, and that is worth its own rung.
    Without it the chain steps straight from "reached the city" to "won the badge", so
    entering the gym pays only `new_map` and walking past it is free -- while actually
    engaging costs `faint` and the HP potential for a fight the agent will probably lose
    the first several times. The gym was rationally ignorable.
    """
    ids = frozenset(M.MAP_IDS[n] for n in names)

    def pred(gs, seen):
        return gs.in_battle == 2 and gs.map_id in ids

    pred.map_ids = ids  # type: ignore[attr-defined]
    return pred


def _gym(badge_key: str, badge_bit: int, gym_map: str, leader: str,
         total_steps: int) -> tuple[Milestone, ...]:
    """The three rungs of a gym: walk in, start a fight, win the badge.

    Every gym previously collapsed to a single "win the badge" milestone with no target
    map. That made a gym rationally ignorable: entering paid only `new_map` and a few
    tiles, walking past cost nothing, and engaging cost `faint` plus the HP potential for
    a fight that is lost the first several times. Viridian Gym was the clearest case --
    the agent crosses Viridian City constantly and never once went in.

    A badge also names no map, so `targets()` fell back to "the deepest map reached" and
    the archive never aimed a single restore at a gym door. Each badge now carries its
    gym explicitly.

    The engaged rung fires on any *trainer* battle on the gym map, not solely the leader
    -- most gyms have junior trainers, and beating them is the intended route to the
    leader anyway. It is named for what it measures rather than for the leader.
    """
    entered, engaged = 400, 300
    return (
        Milestone(f"{badge_key}_gym", f"Entered {leader}'s gym ({gym_map_label(gym_map)})",
                  _visited(gym_map), entered),
        Milestone(f"{badge_key}_engaged", f"Fighting in {gym_map_label(gym_map)}",
                  _fighting_in(gym_map), engaged),
        Milestone(badge_key, f"{_BADGE_NAMES[badge_bit]} ({leader})",
                  _badge(badge_bit), max(total_steps - entered - engaged, 100),
                  target_maps=frozenset({M.MAP_IDS[gym_map]})),
    )


_BADGE_NAMES = (
    "Boulder Badge", "Cascade Badge", "Thunder Badge", "Rainbow Badge",
    "Soul Badge", "Marsh Badge", "Volcano Badge", "Earth Badge",
)


def gym_map_label(gym_map: str) -> str:
    return gym_map.replace("_", " ").title()


def _dex(n: int) -> Predicate:
    return lambda gs, seen: gs.dex_owned >= n


def _event(index: int) -> Predicate:
    """Predicate on a single story flag. Exact and unfakeable, unlike a positional proxy."""
    return lambda gs, seen: gs.has_event(index)


def _all(*preds: Predicate) -> Predicate:
    def pred(gs, seen):
        return all(p(gs, seen) for p in preds)

    pred.map_ids = frozenset().union(*(_map_ids_of(p) for p in preds))  # type: ignore
    return pred


@dataclass(frozen=True)
class Milestone:
    key: str
    label: str
    predicate: Predicate = field(repr=False)
    # Rough number of *agent steps* an expert needs to get here from the previous
    # milestone. Used only to set per-milestone episode budgets and to weight the
    # expected-time bound in the proof; not a reward.
    expert_steps: int = 2_000
    # Maps the agent must be standing on to make progress towards this milestone.
    #
    # Defaults to the maps the predicate itself keys on, which is right for "go here"
    # milestones. It must be given explicitly for milestones defined by a story flag,
    # because a flag names no map -- and for anything requiring a *round trip*. The
    # archive uses this to bias restores towards where the work actually is, which is
    # not always forwards: delivering Oak's Parcel means walking back from Viridian to
    # Pallet Town, and a purely forward-biased archive makes that nearly impossible.
    target_maps: frozenset[int] = frozenset()

    def satisfied(self, gs: GameState, seen_maps: Iterable[int]) -> bool:
        return self.predicate(gs, frozenset(seen_maps))

    def targets(self) -> frozenset[int]:
        return self.target_maps or _map_ids_of(self.predicate)


# --------------------------------------------------------------------------------------
# The chain. Ordering follows the canonical any%-glitchless route.
# --------------------------------------------------------------------------------------

MILESTONES: tuple[Milestone, ...] = (
    Milestone("boot", "In control of the character",
              lambda gs, seen: True, 0),
    Milestone("leave_room", "Left the upstairs bedroom",
              _visited("REDS_HOUSE_1F", "PALLET_TOWN"), 300),
    Milestone("leave_house", "Outside in Pallet Town",
              _visited("PALLET_TOWN"), 300),
    Milestone("oaks_lab", "Entered Oak's Lab",
              _visited("OAKS_LAB"), 600),
    Milestone("got_starter", "Received a starter Pokemon",
              _party(1), 800),
    Milestone("route_1", "On Route 1",
              _visited("ROUTE_1"), 600),
    Milestone("viridian", "Reached Viridian City",
              _visited("VIRIDIAN_CITY"), 800),
    # The Viridian north exit is blocked by an old man until Oak's Parcel is delivered,
    # so these two are a hard gate on the entire rest of the game -- not optional
    # flavour. Both are now verified by the actual story flag rather than by position:
    # the earlier positional proxy fired merely on arriving in Viridian, which hid the
    # fact that the agent had never so much as picked the parcel up.
    #
    # `target_maps` is what tells the archive where the work is. The parcel run is a
    # *round trip* (Viridian Mart, then back to Pallet Town), and it is the case that
    # broke the forward-only archive bias.
    # Target the Mart interior alone, not the city around it. The parcel comes from
    # talking to the clerk, so a restore should land the agent in the same room as him;
    # including VIRIDIAN_CITY put 99.8% of restores in the city (60 cells) and 0% in the
    # Mart (4 cells), which is where the agent had already spent 9M steps getting nowhere.
    Milestone("got_parcel", "Picked up Oak's Parcel from the Viridian Mart",
              _event(RM.EVENT_GOT_OAKS_PARCEL), 1_500,
              frozenset({M.MAP_IDS["VIRIDIAN_MART"]})),
    # Keyed on EVENT_OAK_GOT_PARCEL rather than EVENT_GOT_POKEDEX because that flag was
    # *observed* firing at the exact step the parcel left the bag, whereas GOT_POKEDEX's
    # index had been wrong twice. It is also the semantically correct gate: the old man
    # blocking Viridian's north exit moves once Oak has the parcel.
    #
    # Targets Oak's Lab alone. Including Route 1 and Pallet Town -- "the way there" --
    # sent 88.8% of restores to Route 1 and left the agent rarely starting in the room
    # where the delivery happens. From a lab cell a random policy delivers in ~830 steps.
    Milestone("parcel_returned", "Delivered Oak's Parcel to Oak",
              _event(RM.EVENT_OAK_GOT_PARCEL), 2_500,
              frozenset({M.MAP_IDS["OAKS_LAB"]})),
    Milestone("route_2", "On Route 2",
              _visited("ROUTE_2"), 700),
    Milestone("viridian_forest", "Entered Viridian Forest",
              _visited("VIRIDIAN_FOREST"), 900),
    # The forest's north exit, and the reason it needs its own rung.
    #
    # `MAP_RANK` is derived only from maps a milestone predicate names, so a connector
    # map nobody mentions ranks -1. Measured: the agent reached this gate at 57M env
    # steps and the archive scored those cells 12.6-12.9 against 17.8-18.1 for ordinary
    # Viridian Forest cells -- the states furthest along the path scored *lowest*, and
    # `chosen` was 0 for all three of them across three hours. Every one of 1119
    # restores went back into the forest.
    #
    # Ranking the gate also fixes Route 2, which spans both sides of the forest and
    # takes rank 9 from "On Route 2" (MAP_RANK keeps the earliest milestone that names
    # a map). Emerging north onto Route 2 therefore *lowered* a cell's rank below the
    # forest it had just escaped. A cell that has seen the gate now carries this
    # milestone instead, and milestone dominates rank in `Cell.score`.
    # The 400 steps come *out* of the Pewter rung below rather than being added on top:
    # this splits one journey into two measured halves, so the proof's L is unchanged.
    Milestone("forest_north_gate", "Left Viridian Forest by the north gate",
              _visited("VIRIDIAN_FOREST_NORTH_GATE"), 400),
    Milestone("pewter", "Reached Pewter City",
              _visited("PEWTER_CITY"), 2_100),
    *_gym("badge_1", 0, "PEWTER_GYM", "Brock", 3_000),
    Milestone("route_3", "On Route 3",
              _visited("ROUTE_3"), 900),
    Milestone("mt_moon", "Entered Mt. Moon",
              _visited("MT_MOON_1F"), 800),
    Milestone("mt_moon_cleared", "Through Mt. Moon to Route 4",
              _all(_visited("MT_MOON_B2F"), _visited("ROUTE_4")), 4_000),
    Milestone("cerulean", "Reached Cerulean City",
              _visited("CERULEAN_CITY"), 1_200),
    *_gym("badge_2", 1, "CERULEAN_GYM", "Misty", 3_500),
    Milestone("bills_house", "Met Bill (S.S. Anne ticket)",
              _visited("BILLS_HOUSE"), 3_000),
    Milestone("vermilion", "Reached Vermilion City",
              _visited("VERMILION_CITY"), 2_500),
    Milestone("ss_anne", "Boarded the S.S. Anne",
              _visited("SS_ANNE_1F"), 1_500),
    Milestone("got_cut", "Reached the S.S. Anne captain (HM01 Cut)",
              _visited("SS_ANNE_CAPTAINS_ROOM"), 3_000),
    *_gym("badge_3", 2, "VERMILION_GYM", "Lt. Surge", 4_000),
    Milestone("rock_tunnel", "Entered Rock Tunnel",
              _visited("ROCK_TUNNEL_1F"), 4_000),
    Milestone("lavender", "Reached Lavender Town",
              _visited("LAVENDER_TOWN"), 3_000),
    Milestone("celadon", "Reached Celadon City",
              _visited("CELADON_CITY"), 3_000),
    *_gym("badge_4", 3, "CELADON_GYM", "Erika", 4_000),
    Milestone("rocket_hideout", "Entered the Rocket Hideout",
              _visited("ROCKET_HIDEOUT_B1F"), 2_500),
    Milestone("silph_scope", "Cleared the Rocket Hideout (Silph Scope)",
              _visited("ROCKET_HIDEOUT_B4F"), 4_000),
    Milestone("pokemon_tower", "Cleared Pokemon Tower (Poke Flute)",
              _visited("POKEMON_TOWER_7F"), 4_500),
    Milestone("saffron", "Entered Saffron City",
              _visited("SAFFRON_CITY"), 2_500),
    Milestone("silph_co", "Cleared Silph Co. (Master Ball)",
              _visited("SILPH_CO_11F"), 6_000),
    *_gym("badge_6", 5, "SAFFRON_GYM", "Sabrina", 4_000),
    Milestone("fuchsia", "Reached Fuchsia City",
              _visited("FUCHSIA_CITY"), 3_500),
    *_gym("badge_5", 4, "FUCHSIA_GYM", "Koga", 4_000),
    Milestone("safari_hm", "Cleared the Safari Zone (HM03 Surf / Gold Teeth)",
              _visited("SAFARI_ZONE_SECRET_HOUSE"), 5_000),
    Milestone("cinnabar", "Reached Cinnabar Island",
              _visited("CINNABAR_ISLAND"), 3_000),
    Milestone("mansion_key", "Cleared Pokemon Mansion (Secret Key)",
              _visited("POKEMON_MANSION_B1F"), 4_500),
    *_gym("badge_7", 6, "CINNABAR_GYM", "Blaine", 4_000),
    *_gym("badge_8", 7, "VIRIDIAN_GYM", "Giovanni", 5_000),
    Milestone("victory_road", "Entered Victory Road",
              _visited("VICTORY_ROAD_1F"), 3_000),
    Milestone("victory_road_cleared", "Cleared Victory Road",
              _visited("INDIGO_PLATEAU_LOBBY"), 5_000),
    Milestone("lorelei", "Defeated Lorelei",
              _visited("BRUNOS_ROOM"), 3_000),
    Milestone("bruno", "Defeated Bruno",
              _visited("AGATHAS_ROOM"), 2_500),
    Milestone("agatha", "Defeated Agatha",
              _visited("LANCES_ROOM"), 2_500),
    Milestone("lance", "Defeated Lance",
              _visited("CHAMPIONS_ROOM"), 2_500),
    Milestone("hall_of_fame", "HALL OF FAME -- game complete",
              _visited("HALL_OF_FAME"), 2_000),
)

NUM_MILESTONES = len(MILESTONES)
MILESTONE_INDEX: dict[str, int] = {m.key: i for i, m in enumerate(MILESTONES)}
TERMINAL_MILESTONE = NUM_MILESTONES - 1

# map id -> position along the critical path, derived from the chain above so there is
# exactly one source of truth. A map that several milestones reference takes the earliest.
MAP_RANK: dict[int, int] = {}
for _i, _m in enumerate(MILESTONES):
    for _map_id in _map_ids_of(_m.predicate):
        MAP_RANK.setdefault(_map_id, _i)
del _i, _m, _map_id


def chain_fingerprint() -> str:
    """Identity of the milestone chain.

    `best_milestone` is a monotone counter whose meaning is defined by the chain. When
    the chain changes -- milestones inserted, predicates tightened -- a counter carried
    across from a checkpoint silently refers to different milestones. That happened
    concretely: a resumed run held best_milestone=8 from the old chain and pointed the
    archive at delivering Oak's Parcel before the agent had picked it up.
    """
    import hashlib

    h = hashlib.blake2b(digest_size=8)
    for m in MILESTONES:
        h.update(m.key.encode())
    return h.hexdigest()


def achieved_milestone(index: int) -> Milestone | None:
    """The last milestone actually completed, for a prefix count of `index`.

    `MilestoneTracker.index` counts how many consecutive milestones are satisfied, so
    index 6 means MILESTONES[0..5] are done and MILESTONES[6] is the *next target*.
    Indexing the list directly with the count therefore names something that has not
    happened yet -- which is exactly the mistake the run logs made, reporting "Reached
    Viridian City" when the agent had only finished Route 1 and had never set foot in
    Viridian.
    """
    if index <= 0:
        return None
    return MILESTONES[min(index, NUM_MILESTONES) - 1]


def next_milestone(index: int) -> Milestone | None:
    """The milestone currently being worked towards; None once the chain is complete."""
    if index >= NUM_MILESTONES:
        return None
    return MILESTONES[index]


def map_rank(map_id: int) -> int:
    """How far along the critical path a map sits; -1 if it is not on the path.

    Unlike the milestone index, this is a property of the *state itself* and cannot be
    inflated by backtracking, which is what makes it a sound tie-break when ranking
    archived cells (see `FrontierArchive.deepest`).
    """
    return MAP_RANK.get(int(map_id), -1)

# Expected expert step budget for the whole game; the proof uses this as the L in the
# O(L) mixing bound.
TOTAL_EXPERT_STEPS = sum(m.expert_steps for m in MILESTONES)


class MilestoneTracker:
    """Monotone milestone counter for one run (not one episode).

    `index` is the number of *consecutive* milestones satisfied from the start, which is
    the quantity the archive uses to rank cells. We also keep the full satisfied set,
    because a few milestones are legitimately achievable out of order (badge 5 and 6 are
    interchangeable, for instance) and we do not want to lose that credit.
    """

    def __init__(self) -> None:
        self.seen_maps: set[int] = set()
        self.satisfied: set[str] = set()
        self.index: int = 0

    def update(self, gs: GameState) -> list[str]:
        """Returns the keys newly satisfied by this state."""
        self.seen_maps.add(gs.map_id)
        newly: list[str] = []
        seen = frozenset(self.seen_maps)
        for m in MILESTONES:
            if m.key in self.satisfied:
                continue
            if m.predicate(gs, seen):
                self.satisfied.add(m.key)
                newly.append(m.key)
        # Recompute the consecutive prefix length.
        idx = 0
        for m in MILESTONES:
            if m.key in self.satisfied:
                idx += 1
            else:
                break
        self.index = idx
        return newly

    @property
    def achieved_label(self) -> str:
        """What has actually been completed. Use this in any progress report."""
        m = achieved_milestone(self.index)
        return m.label if m is not None else "(nothing yet)"

    @property
    def frontier_label(self) -> str:
        """What is being worked towards next -- deliberately *not* an achievement."""
        m = next_milestone(self.index)
        return m.label if m is not None else "GAME COMPLETE"

    @property
    def completed(self) -> bool:
        return "hall_of_fame" in self.satisfied

    def state_dict(self) -> dict:
        return {
            "seen_maps": sorted(self.seen_maps),
            "satisfied": sorted(self.satisfied),
            "index": self.index,
        }

    def load_state_dict(self, d: dict) -> None:
        self.seen_maps = set(d.get("seen_maps", []))
        self.satisfied = set(d.get("satisfied", []))
        self.index = int(d.get("index", 0))


def milestone_index_of(gs: GameState, seen_maps: Iterable[int]) -> int:
    """Stateless prefix-length evaluation, used when scoring an archived cell."""
    seen = frozenset(seen_maps) | {gs.map_id}
    idx = 0
    for m in MILESTONES:
        if m.predicate(gs, seen):
            idx += 1
        else:
            break
    return idx
