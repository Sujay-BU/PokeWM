"""The subgoal vocabulary.

A closed, finite vocabulary rather than free-form text, for three reasons:

1. It is one-hot encodable, so it can condition the world model's encoder and the actor
   directly (`obs["subgoal"]`) with no text encoder in the loop.
2. Each subgoal carries a *machine-checkable* satisfaction predicate over `GameState`.
   That turns the LLM's suggestion into a verifiable reward event rather than an opinion
   -- if the model proposes HEAL and the party is healed, the bonus pays; if it proposes
   nonsense, nothing pays and the agent is unaffected.
3. A closed set is what makes the safety argument in docs/PROOF.md §5 work: the LLM can
   only reweight exploration among a fixed set of behaviours whose satisfaction
   conditions are all *progress-monotone or neutral*, so a badly-behaved or unavailable
   LLM can slow the agent down but cannot steer it away from the optimum.

This is the "semantic subgoal proposer" pattern from the 2026 LLM-guided RL literature
(STO-RL, MIRA), specialised so that verification is exact instead of learned.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..emulator import maps as M
from ..emulator.ram_map import GameState

# A predicate decides whether the subgoal was achieved between `before` and `after`.
Check = Callable[[GameState, GameState], bool]


def _entered_new_map(before: GameState, after: GameState) -> bool:
    return after.map_id != before.map_id


def _moved(before: GameState, after: GameState) -> bool:
    return after.position != before.position


@dataclass(frozen=True)
class Subgoal:
    id: int
    name: str
    description: str  # shown to the LLM
    check: Check


def _sg(id_: int, name: str, desc: str, check: Check) -> Subgoal:
    return Subgoal(id_, name, desc, check)


# Referenced directly by the env, which forces it when the party is hurt.
HEAL_SUBGOAL_ID = 13
CATCH_SUBGOAL_ID = 11
# Forced when the party is small and the bag holds no balls, which makes CATCH
# unsatisfiable. See `PokemonRedEnv._active_subgoal`.
BUY_BALLS_SUBGOAL_ID = 14

SUBGOALS: tuple[Subgoal, ...] = (
    _sg(0, "EXPLORE",
        "Wander into parts of the current map you have not stood on yet.",
        _moved),
    _sg(1, "LEAVE_AREA",
        "Find an exit, door, or map edge and transition to a different map.",
        _entered_new_map),
    _sg(2, "GO_NORTH", "Travel north/up until the map changes.", _entered_new_map),
    _sg(3, "GO_SOUTH", "Travel south/down until the map changes.", _entered_new_map),
    _sg(4, "GO_EAST", "Travel east/right until the map changes.", _entered_new_map),
    _sg(5, "GO_WEST", "Travel west/left until the map changes.", _entered_new_map),
    _sg(6, "ENTER_BUILDING",
        "Step onto a doorway tile to go inside a building.",
        lambda b, a: _entered_new_map(b, a)),
    _sg(7, "TALK_TO_NPC",
        "Face a person and press A to talk; story flags usually advance.",
        lambda b, a: a.event_flag_bits > b.event_flag_bits),
    # Verified on a story-flag advance rather than on a text box closing: `wTextBoxID`
    # is nonzero even in the plain overworld, so the obvious predicate
    # (`b.text_box_id != 0 and a.text_box_id == 0`) could never fire and this subgoal
    # was silently unrewardable.
    _sg(8, "ADVANCE_DIALOG",
        "Press A to clear the text box currently on screen.",
        lambda b, a: a.event_flag_bits > b.event_flag_bits or (
            b.joy_ignore != 0 and a.joy_ignore == 0
        )),
    _sg(9, "WIN_BATTLE",
        "Attack until the opposing Pokemon faints.",
        lambda b, a: b.in_battle != 0 and a.in_battle == 0 and not a.party_wiped),
    _sg(10, "FLEE_BATTLE",
        "Run away from the current wild battle.",
        lambda b, a: b.in_battle == 1 and a.in_battle == 0),
    _sg(CATCH_SUBGOAL_ID, "CATCH_POKEMON",
        "Weaken a wild Pokemon and throw a Poke Ball.",
        lambda b, a: a.party_size > b.party_size or a.dex_owned > b.dex_owned),
    _sg(12, "TRAIN_LEVELS",
        "Fight wild Pokemon in tall grass to raise party levels.",
        lambda b, a: a.party_level_sum > b.party_level_sum),
    _sg(HEAL_SUBGOAL_ID, "HEAL",
        "Go to a Pokemon Center counter and talk to the nurse.",
        lambda b, a: a.party_hp_frac > b.party_hp_frac + 1e-6),
    # Satisfied by *balls* specifically, not by any bag slot. `bag_item_count` counts
    # distinct slots, so buying a second Potion into an existing stack changes nothing
    # and buying an Antidote satisfied a subgoal issued to enable catching.
    _sg(BUY_BALLS_SUBGOAL_ID, "BUY_ITEMS",
        "Use a Poke Mart counter to buy Poke Balls.",
        lambda b, a: a.ball_count > b.ball_count),
    _sg(15, "USE_ITEM",
        "Open the START menu, choose ITEM, and use something.",
        lambda b, a: a.bag_item_count < b.bag_item_count),
    _sg(16, "CHALLENGE_GYM",
        "Enter the local Gym and defeat the Gym Leader for a badge.",
        lambda b, a: a.badge_count > b.badge_count),
    _sg(17, "USE_FIELD_MOVE",
        "Use CUT, SURF, or STRENGTH from the party menu to clear an obstacle.",
        lambda b, a: a.walk_bike_surf != b.walk_bike_surf or _entered_new_map(b, a)),
    _sg(18, "SOLVE_PUZZLE",
        "Push boulders or flip switches to open the way forward.",
        lambda b, a: a.event_flag_bits > b.event_flag_bits),
    _sg(19, "BACKTRACK",
        "Return to a previously visited map to pick up something missed.",
        _entered_new_map),
    _sg(20, "REACH_NEXT_CITY",
        "Follow routes onward to the next town or city on the main path.",
        lambda b, a: a.map_id in M.CITY_ORDER and a.map_id != b.map_id),
    _sg(21, "ORGANIZE_PARTY",
        "Open the START menu and reorder or inspect Pokemon.",
        lambda b, a: a.menu_item != b.menu_item),
    _sg(22, "MAIN_QUEST",
        "Do whatever the story currently asks; advance any story flag.",
        lambda b, a: a.event_flag_bits > b.event_flag_bits),
    _sg(23, "NONE",
        "No specific guidance; let the learned policy decide.",
        lambda b, a: False),
)

NUM_SUBGOALS = len(SUBGOALS)
BY_NAME: dict[str, Subgoal] = {s.name: s for s in SUBGOALS}
BY_ID: dict[int, Subgoal] = {s.id: s for s in SUBGOALS}
DEFAULT_SUBGOAL = BY_NAME["MAIN_QUEST"].id

assert [s.id for s in SUBGOALS] == list(range(NUM_SUBGOALS)), "subgoal ids must be dense"


def parse_subgoal(name: str | None) -> int:
    """Map an LLM's answer onto the vocabulary, tolerantly.

    Anything unrecognised becomes MAIN_QUEST, which is the identity-ish option: it pays
    out on any story-flag advance, i.e. exactly the extrinsic objective. A malformed LLM
    response therefore degrades to "no guidance", never to bad guidance.
    """
    if not name:
        return DEFAULT_SUBGOAL
    key = name.strip().upper().replace(" ", "_").replace("-", "_")
    if key in BY_NAME:
        return BY_NAME[key].id
    for candidate in BY_NAME:
        if candidate in key:
            return BY_NAME[candidate].id
    return DEFAULT_SUBGOAL


def vocabulary_prompt() -> str:
    return "\n".join(f"- {s.name}: {s.description}" for s in SUBGOALS)


def satisfied(subgoal_id: int, before: GameState, after: GameState) -> bool:
    sg = BY_ID.get(subgoal_id)
    if sg is None:
        return False
    try:
        return bool(sg.check(before, after))
    except Exception:
        return False
