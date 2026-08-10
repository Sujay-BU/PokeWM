"""Scripted traversal of the title / naming screens.

Everything before the player gains control is a fixed, information-free prefix: a logo,
a menu, two name-entry screens and ~90 dialogue boxes. It contains no decision an agent
could learn anything from, and the name-entry keyboard alone is a 200-step combinatorial
detour that would dominate early training. Every published Pokemon Red agent
(PokemonRedExperiments, pokegym, PokeRL) starts from a canned post-intro save state for
this reason.

We do not ship one. Instead this module *derives* it from the ROM on first run using
only button presses and RAM reads, so the artifact is reproducible and auditable. The
learned policy's episode begins exactly where control begins.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

from . import maps as M
from . import ram_map as RM

log = logging.getLogger(__name__)

# Number of emulator ticks a bootstrap button press is held / released for. These are
# generous compared to the training env because we are optimising for robustness across
# variable-length text boxes, not throughput.
_PRESS = 10
_RELEASE = 10

# The naming menus ("NEW NAME / RED / ASH / JACK") are the only 4-entry menus reachable
# during the intro; the title menu has at most 3 entries.
_NAME_MENU_MAX_ITEM = 3


def _tap(pyboy, button: str, press: int = _PRESS, release: int = _RELEASE) -> None:
    pyboy.button_press(button)
    pyboy.tick(press, False)
    pyboy.button_release(button)
    pyboy.tick(release, False)


def _in_control(reader: RM.RamReader) -> bool:
    """True once the player sprite accepts input in the bedroom.

    Three conditions together are reliable: the bedroom map is loaded, no script owns
    the joypad, and no text box is on screen.
    """
    gs_map = reader.byte(RM.CUR_MAP)
    joy = reader.byte(RM.JOY_IGNORE)
    return gs_map == M.MAP_IDS["REDS_HOUSE_2F"] and joy == 0


def _confirm_movable(pyboy, reader: RM.RamReader) -> bool:
    """Verify control by actually moving and checking the coordinate changed."""
    before = (reader.byte(RM.X_COORD), reader.byte(RM.Y_COORD))
    for button in ("down", "up", "left", "right"):
        _tap(pyboy, button, press=16, release=16)
        after = (reader.byte(RM.X_COORD), reader.byte(RM.Y_COORD))
        if after != before:
            return True
    return False


def run_intro(pyboy, max_taps: int = 900) -> bool:
    """Drive the emulator from power-on to player control.

    Returns True on success. Leaves the emulator sitting in Red's bedroom.
    """
    reader = RM.RamReader(pyboy.memory)

    # Boot logos. ~4 s of game time before the title screen accepts input.
    pyboy.tick(400, False)

    settled = 0
    for tap in range(max_taps):
        if _in_control(reader):
            # Require the condition to hold for several consecutive taps: it flickers
            # true for a frame or two during the final fade-in.
            settled += 1
            if settled >= 3 and _confirm_movable(pyboy, reader):
                log.info("intro complete after %d taps", tap)
                return True
        else:
            settled = 0

        # A 4-entry menu during the intro is always a name-selection screen. Pick the
        # preset name (RED / BLUE) rather than entering the text keyboard.
        if reader.byte(RM.MAX_MENU_ITEM) >= _NAME_MENU_MAX_ITEM:
            _tap(pyboy, "down")
            _tap(pyboy, "a")
            continue

        # Otherwise advance whatever is on screen. START dismisses the title, A
        # advances dialogue and confirms the NEW GAME menu entry.
        _tap(pyboy, "start" if tap < 4 else "a")

    return False


def make_init_state(
    rom_path: str | Path,
    out_path: str | Path,
    *,
    render: bool = False,
    overwrite: bool = False,
) -> Path:
    """Produce (and cache) the post-intro save state."""
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from pyboy import PyBoy

    pyboy = PyBoy(
        str(rom_path),
        window="SDL2" if render else "null",
        sound_emulated=False,
        log_level="ERROR",
    )
    try:
        pyboy.set_emulation_speed(1 if render else 0)
        if not run_intro(pyboy):
            raise RuntimeError(
                "Could not script through the Pokemon Red intro. Re-run with "
                "render=True to watch where it stalls."
            )
        reader = RM.RamReader(pyboy.memory)
        gs = reader.read()
        if gs.map_id != M.MAP_IDS["REDS_HOUSE_2F"]:
            raise RuntimeError(f"unexpected post-intro map 0x{gs.map_id:02X}")

        buf = io.BytesIO()
        pyboy.save_state(buf)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_bytes(buf.getvalue())
        tmp.replace(out_path)
        log.info("wrote init state to %s (%d bytes)", out_path, out_path.stat().st_size)
        return out_path
    finally:
        pyboy.stop(save=False)


def ensure_init_state(cfg) -> str:
    """Idempotent helper used by the env and the trainer."""
    target = Path(cfg.save_state_dir) / "post_intro.state"
    make_init_state(cfg.rom_path, target)
    return str(target)
