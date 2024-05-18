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

    def flipy(self) -> Pattern:
        return Pattern(np.flip(self.filled, 0))

    def flipx(self) -> Pattern:
        return Pattern(np.flip(self.filled, 1))

    def place(
        self,
        offset: tuple[int, int] | None = None,
        corner_offset: tuple[int, int] | None = None,
    ) -> Positions:
        """Return positions for the pattern.

        Parameters
        ----------
        offset:
            Shift the centre of the pattern byt this offset.
        """
        size = self.filled.shape
        y, x = np.where(self.filled)
        if offset is not None:
            x = x + offset[1] - size[1] // 2
            y = y + offset[0] - size[0] // 2
        if corner_offset is not None:
            x = x + corner_offset[1]
            y = y + corner_offset[0]

        return Positions(
            x=x,
            y=y,
        )

    @property
    def cost(self) -> int:
        return np.sum(self.filled)

    @property
    def shape(self) -> tuple[int, int]:
        return self.filled.shape


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
    n_col = max(len(row) for row in filled)
    filled = [row + [False] * (n_col - len(row)) for row in filled]
    filled = filled[::-1]  # flip y to match coord system of game
    filled = np.array(filled)
    return Pattern(filled)
