# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional, Tuple

import numpy as np
from cholerama import Positions

from .patterns import Pattern, load_pattern

AUTHOR = "Protomolecule"  # This is your team name
SEED = None  # Set this to a value to make runs reproducible


class Bot:
    """
    This is the bot that will be instantiated for the competition.

    The pattern can be either a numpy array or a path to an image (white means 0,
    black means 1).
    """

    def __init__(
        self,
        number: int,
        name: str,
        patch_location: Tuple[int, int],
        patch_size: Tuple[int, int],
    ):
        """
        Parameters:
        ----------
        number: int
            The player number. Numbers on the board equal to this value mark your cells.
        name: str
            The player's name
        patch_location: tuple
            The i, j row and column indices of the patch in the grid
        patch_size: tuple
            The size of the patch
        """
        self.number = number  # Mandatory: this is your number on the board
        self.name = name  # Mandatory: player name
        self.color = "#5fa5ad"
        self.patch_location = patch_location
        self.patch_size = patch_size

        self.rng = np.random.default_rng(SEED)

        self.patterns = load_patterns()

        # Spawn initial spaceship to stay alive until we have enough tokens.
        self.pattern = self.patterns["glider"].rotate(2).place_centre()
        self.pending_pattern: Pattern | None = self.patterns["max107"]
        #
        # bl = pattern.rotate(0).place_centre(
        #     offset=(patch_size[0] // 4, patch_size[1] // 4)
        # )
        # tl = pattern.rotate(1).place_centre(
        #     offset=(patch_size[0] // 4, 3 * patch_size[1] // 4)
        # )
        # self.pattern = merge_positions(bl, tl)
        # tr = pattern.rotate(2).place_centre(
        #     offset=(3 * patch_size[0] // 4, 3 * patch_size[1] // 4)
        # )
        # br = pattern.rotate(3).place_centre(
        #     offset=(3 * patch_size[0] // 4, patch_size[1] // 4)
        # )
        # self.pattern = bl
        # self.pattern = merge_positions(merge_positions(merge_positions(bl, tl), tr), br)

        # If we make the pattern too sparse, it just dies quickly
        # xy = self.rng.integers(0, 12, size=(2, 100))
        # self.pattern = Positions(
        #     x=xy[1] + patch_size[1] // 2, y=xy[0] + patch_size[0] // 2
        # )
        # The pattern can also be just an image (0=white, 1=black)
        # self.pattern = "mypattern.png"

    def iterate(
        self, iteration: int, board: np.ndarray, patch: np.ndarray, tokens: int
    ) -> Optional[Positions]:
        """
        This method will be called by the game engine on each iteration.

        Parameters:
        ----------
        iteration : int
            The current iteration number.
        board : numpy array
            The current state of the entire board.
        patch : numpy array
            The current state of the player's own patch on the board.
        tokens : list
            The list of tokens on the board.

        Returns:
        -------
        An object containing the x and y coordinates of the new cells.
        """
        if self.pending_pattern is not None and tokens >= self.pending_pattern.cost:
            pattern = self.pending_pattern
            self.pending_pattern = None
            return pattern.place_centre((patch.shape[0] // 2, patch.shape[1] // 2))


def merge_positions(*pos: Positions) -> Positions:
    return Positions(
        x=np.concatenate([p.x for p in pos]),
        y=np.concatenate([p.y for p in pos]),
    )


_PATTERN_NAMES = [
    "backrake2",
    "blocklayingswitchenginepredecessor",
    "glider",
    "lwss",
    "max107",
    "max110",
    "max127",
    "timebomb",
]


def load_patterns() -> dict[str, Pattern]:
    return {name: load_pattern(name) for name in _PATTERN_NAMES}
