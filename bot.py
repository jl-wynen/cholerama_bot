# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional, Tuple

import numpy as np
from cholerama import Positions, helpers

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
        match self.rng.integers(0, 3):
            case 0:
                self.strategy = GuardedBomb(self.rng, self.patch_size)
            case 1:
                self.strategy = EarlyBomb(self.rng, self.patch_size)
            case 2:
                self.strategy = LateBomb(self.rng, self.patch_size)
        self.pattern = self.strategy.place_initial()

    def iterate(
        self, iteration: int, board: np.ndarray, patch: np.ndarray, tokens: int
    ) -> Optional[Positions]:
        """
        This method will be called by the game engine on each iteration.

        Parameters:
        ----------
        iteration:
            The current iteration number.
        board:
            The current state of the entire board.
        patch:
            The current state of the player's own patch on the board.
        tokens:
            Number of available tokens.

        Returns:
        -------
        An object containing the x and y coordinates of the new cells.
        """
        return self.strategy.iterate(iteration, board, patch, tokens)


class EarlyBomb:
    def __init__(self, rng, patch_size):
        self.rng = rng
        self.patch_size = patch_size
        self.lwss = load_pattern("lwss")
        self.pending_pattern: Pattern | None = load_pattern("max107")

    def place_initial(self) -> Positions:
        match self.rng.integers(0, 4):
            case 0:
                rot = self.rng.choice([0, 3])
                a, b = 1, 1
            case 1:
                rot = self.rng.choice([0, 1])
                a, b = 3, 1
            case 2:
                rot = self.rng.choice([1, 2])
                a, b = 3, 3
            case 3:
                rot = self.rng.choice([2, 3])
                a, b = 1, 3
            case _:
                rot = 0
                a, b = 1, 1
        return self.lwss.rotate(rot).place(
            (a * self.patch_size[0] // 4, b * self.patch_size[1] // 4)
        )

    def iterate(
        self, iteration: int, board: np.ndarray, patch: np.ndarray, tokens: int
    ) -> Positions | None:
        if self.pending_pattern is not None:
            if tokens >= self.pending_pattern.cost:
                pattern = self.pending_pattern
                self.pending_pattern = None
                return pattern.place((patch.shape[0] // 2, patch.shape[1] // 2))
            else:
                return

        if tokens >= self.lwss.cost:
            empty_regions = helpers.find_empty_regions(patch, self.lwss.shape)
            n_regions = len(empty_regions)
            if n_regions == 0:
                return
            offset = empty_regions[self.rng.integers(0, n_regions)]
            if offset[0] < patch.shape[0] // 2 and offset[1] < patch.shape[1] // 2:
                return self.lwss.rotate(self.rng.choice([0, 3])).place(
                    corner_offset=offset
                )
            elif offset[0] > patch.shape[0] // 2 and offset[1] < patch.shape[1] // 2:
                return self.lwss.rotate(self.rng.choice([0, 1])).place(
                    corner_offset=offset
                )
            elif offset[0] > patch.shape[0] // 2 and offset[1] > patch.shape[1] // 2:
                return self.lwss.rotate(self.rng.choice([1, 2])).place(
                    corner_offset=offset
                )
            elif offset[0] < patch.shape[0] // 2 and offset[1] > patch.shape[1] // 2:
                return self.lwss.rotate(self.rng.choice([2, 3])).place(
                    corner_offset=offset
                )


class GuardedBomb(EarlyBomb):
    def __init__(self, rng, patch_size):
        super().__init__(rng, patch_size)
        self.backrake2 = load_pattern("backrake2")

    def place_initial(self) -> Positions:
        match self.rng.integers(0, 4):
            case 0:
                a = self.backrake2.rotate(0).place(
                    (1 * self.patch_size[0] // 4, 1 * self.patch_size[1] // 4)
                )
                b = self.backrake2.rotate(2).place(
                    (3 * self.patch_size[0] // 4, 3 * self.patch_size[1] // 4)
                )
            case 1:
                a = self.backrake2.rotate(1).place(
                    (3 * self.patch_size[0] // 4, 1 * self.patch_size[1] // 4)
                )
                b = self.backrake2.rotate(3).place(
                    (1 * self.patch_size[0] // 4, 3 * self.patch_size[1] // 4)
                )
            case 2:
                a = self.backrake2.rotate(2).place(
                    (3 * self.patch_size[0] // 4, 3 * self.patch_size[1] // 4)
                )
                b = self.backrake2.rotate(1).place(
                    (1 * self.patch_size[0] // 4, 1 * self.patch_size[1] // 4)
                )
            case 3:
                a = self.backrake2.rotate(3).place(
                    (1 * self.patch_size[0] // 4, 3 * self.patch_size[1] // 4)
                )
                b = self.backrake2.rotate(1).place(
                    (3 * self.patch_size[0] // 4, 1 * self.patch_size[1] // 4)
                )
            case _:
                raise RuntimeError("asd")
        return merge_positions(a, b)


class LateBomb:
    def __init__(self, rng, patch_size):
        self.rng = rng
        self.patch_size = patch_size
        self.lwss = load_pattern("lwss")
        self.pending_pattern: Pattern | None = load_pattern("max107")
        self.backrake2 = load_pattern("backrake2")

    def place_initial(self) -> Positions:
        a, b = 0, 0
        while a == b:
            a, b = self.rng.integers(0, 4, 2)

        return merge_positions(self.place_backrake2(a), self.place_backrake2(b))

    def place_backrake2(self, i: int) -> Positions:
        match i:
            case 0:
                return (
                    self.backrake2.rotate(0)
                    .flipy()
                    .place((1 * self.patch_size[0] // 4, 1 * self.patch_size[1] // 4))
                )
            case 1:
                return self.backrake2.rotate(2).place(
                    (1 * self.patch_size[0] // 4, 3 * self.patch_size[1] // 4)
                )
            case 2:
                return self.backrake2.rotate(1).place(
                    (3 * self.patch_size[0] // 4, 3 * self.patch_size[1] // 4)
                )
            case 3:
                return (
                    self.backrake2.rotate(3)
                    .flipy()
                    .place((3 * self.patch_size[0] // 4, 1 * self.patch_size[1] // 4))
                )

    def iterate(
        self, iteration: int, board: np.ndarray, patch: np.ndarray, tokens: int
    ) -> Positions | None:
        if iteration < 1500:
            if tokens >= self.backrake2.cost:
                return self.place_backrake2(self.rng.integers(0, 4))
        else:
            if self.pending_pattern is not None:
                if tokens >= self.pending_pattern.cost:
                    pattern = self.pending_pattern
                    self.pending_pattern = None
                    return pattern.place((patch.shape[0] // 2, patch.shape[1] // 2))
                else:
                    return

            if tokens >= self.lwss.cost:
                empty_regions = helpers.find_empty_regions(patch, self.lwss.shape)
                n_regions = len(empty_regions)
                if n_regions == 0:
                    return
                offset = empty_regions[self.rng.integers(0, n_regions)]
                if offset[0] < patch.shape[0] // 2 and offset[1] < patch.shape[1] // 2:
                    return self.lwss.rotate(self.rng.choice([0, 3])).place(
                        corner_offset=offset
                    )
                elif (
                    offset[0] > patch.shape[0] // 2 and offset[1] < patch.shape[1] // 2
                ):
                    return self.lwss.rotate(self.rng.choice([0, 1])).place(
                        corner_offset=offset
                    )
                elif (
                    offset[0] > patch.shape[0] // 2 and offset[1] > patch.shape[1] // 2
                ):
                    return self.lwss.rotate(self.rng.choice([1, 2])).place(
                        corner_offset=offset
                    )
                elif (
                    offset[0] < patch.shape[0] // 2 and offset[1] > patch.shape[1] // 2
                ):
                    return self.lwss.rotate(self.rng.choice([2, 3])).place(
                        corner_offset=offset
                    )


# This does not find a match!
# def find_matching_position(
#     pattern: Pattern,
#     board: np.ndarray,
#     max_diff: int,
# ) -> tuple[int, int] | None:
#     mi = 100
#     for y in range(board.shape[0] - pattern.shape[0]):
#         for x in range(board.shape[1] - pattern.shape[1]):
#             region = board[y : y + pattern.shape[0], x : x + pattern.shape[1]]
#             diff = np.sum(np.abs(region != pattern.filled))
#             if diff < mi:
#                 mi = diff
#             if diff <= max_diff:
#                 print("found", x, y)
#                 return x, y
#     print("none", mi)
#     return None


def merge_positions(*pos: Positions) -> Positions:
    return Positions(
        x=np.concatenate([p.x for p in pos]),
        y=np.concatenate([p.y for p in pos]),
    )


_PATTERN_NAMES = [
    "backrake2",
    "blocklayingswitchenginepredecessor",
    "box",
    "glider",
    "lwss",
    "max107",
    "max110",
    "max127",
    "timebomb",
]


def load_patterns() -> dict[str, Pattern]:
    return {name: load_pattern(name) for name in _PATTERN_NAMES}
