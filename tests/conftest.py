"""Shared fixtures.

Tests are split into two tiers:

* pure-logic tests, which construct fake RAM and fake tensors and run in milliseconds;
* `@pytest.mark.emulator` tests, which boot a real PyBoy on the real ROM.

The emulator tier is skipped automatically when the ROM is absent, so the suite stays
runnable in a checkout without the cartridge dump.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokewm.config import Config
from pokewm.emulator import ram_map as RM


class FakeMemory:
    """Mimics PyBoy's `pyboy.memory`: scalar and slice indexing over 64 KiB."""

    def __init__(self, size: int = 0x10000) -> None:
        self.data = bytearray(size)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return list(self.data[key])
        return self.data[key]

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            self.data[key] = bytes(value)
        else:
            self.data[key] = value & 0xFF

    # -- helpers used by tests -------------------------------------------------

    def write(self, addr: int, values) -> None:
        if isinstance(values, int):
            self.data[addr] = values & 0xFF
        else:
            self.data[addr : addr + len(values)] = bytes(values)

    def write_u16_be(self, addr: int, value: int) -> None:
        self.data[addr] = (value >> 8) & 0xFF
        self.data[addr + 1] = value & 0xFF


def make_party_mon(mem: FakeMemory, slot: int, species: int, level: int, hp: int,
                   max_hp: int, status: int = 0) -> None:
    base = RM.PARTY_MON_1 + slot * RM.PARTY_MON_STRIDE
    mem.write(base + RM.OFF_SPECIES, species)
    mem.write_u16_be(base + RM.OFF_HP, hp)
    mem.write_u16_be(base + RM.OFF_MAX_HP, max_hp)
    mem.write(base + RM.OFF_LEVEL, level)
    mem.write(base + RM.OFF_STATUS, status)


@pytest.fixture
def memory() -> FakeMemory:
    return FakeMemory()


@pytest.fixture
def basic_state(memory: FakeMemory) -> RM.GameState:
    """A plausible mid-early-game state."""
    memory.write(RM.CUR_MAP, 0x03)  # Cerulean City
    memory.write(RM.X_COORD, 20)
    memory.write(RM.Y_COORD, 30)
    memory.write(RM.PARTY_COUNT, 2)
    make_party_mon(memory, 0, species=4, level=22, hp=30, max_hp=60)
    make_party_mon(memory, 1, species=16, level=18, hp=44, max_hp=44)
    memory.write(RM.BADGES, 0b00000011)  # boulder + cascade
    memory.write(RM.MONEY, [0x00, 0x12, 0x34])  # $1234 BCD
    memory.write(RM.POKEDEX_OWNED, [0xFF, 0x03] + [0] * 17)  # 10 bits set
    memory.write(RM.POKEDEX_SEEN, [0xFF, 0xFF] + [0] * 17)  # 16 bits set
    memory.write(RM.EVENT_FLAGS_START, [0xFF] * 4)  # 32 story flags
    return RM.RamReader(memory).read()


@pytest.fixture
def smoke_config() -> Config:
    return Config.preset("smoke")


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "emulator: needs the real ROM and boots PyBoy (slow)"
    )
    config.addinivalue_line("markers", "slow: takes more than a few seconds")
    config.addinivalue_line("markers", "llm: needs a running Ollama daemon")
