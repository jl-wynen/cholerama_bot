# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import importlib.resources
from dataclasses import dataclass

import numpy as np
from cholerama import Positions


@dataclass
class Pattern:
    filled: np.ndarray

    def rotate(self, k: int) -> Pattern:
        return Pattern(np.rot90(self.filled, k))

    def place_centre(self, offset: tuple[int, int] | None = None) -> Positions:
        size = self.filled.shape
        y, x = np.where(self.filled)
        if offset is not None:
            x = x + offset[0] - size[0] // 2
            y = y + offset[1] - size[1] // 2

        return Positions(
            x=x,
            y=y,
        )


def load_pattern(name: str) -> Pattern:
    cells = (
        importlib.resources.files("protomolecule_bot.patterns")
        .joinpath(f"{name}.cells")
        .read_text(encoding="utf-8")
    )
    filled = []
    for line in cells.splitlines():
        if line.startswith("!"):
            continue
        filled.append([c == "O" for c in line])
    filled = np.array(filled)
    return Pattern(filled)
