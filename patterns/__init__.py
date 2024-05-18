# SPDX-License-Identifier: BSD-3-Clause
import importlib.resources

import numpy as np
from cholerama import Positions


def load_pattern(name: str) -> Positions:
    cells = (
        importlib.resources.files("protomolecule_bot.patterns")
        .joinpath(f"{name}.cells")
        .read_text(encoding="utf-8")
    )
    x = []
    y = []
    row = 0
    for line in cells.splitlines():
        if line.startswith("!"):
            continue
        for col, cell in enumerate(line):
            print(row, col, cell)
            if cell == "O":
                x.append(col)
                y.append(row)
        row += 1
    return Positions(x=np.array(x), y=np.array(y))
