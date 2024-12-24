"""Connect Four game for mulitplayer Reinforcement Learning trial."""

from typing import TypedDict

from numpy import count_nonzero, diagonal, int_, ndarray, zeros
from numpy.lib.stride_tricks import sliding_window_view


class MoveStatus(TypedDict):
    """Report the game status after `Game.step()`."""
    illegal: bool
    win: bool
    full: bool


class Game:
    """Connect Four game."""

    def __init__(self) -> None:
        # TODO(Jean): Variablize board size (Fanny's suggestion)
        self.board = zeros((7, 7), dtype=int_)

    def observe(self) -> ndarray:
        """Read the current board state."""
        return self.board

    def step(self, action: dict[str, int]) -> MoveStatus:
        """Play a single move."""
        print(action)
        player_id = action["player_id"]
        column = action["column"]

        height = count_nonzero(self.board[column, :])
        if height == self.board[column, :].size:
            return MoveStatus(illegal=True, win=False, full=False)
        self.board[column, height] = player_id

        win = self.is_win(column, height)
        full = (self.board == 0).sum() == 0
        return MoveStatus(illegal=False, win=win, full=full)

    def is_win(self, column, height) -> bool:
        """Verify is the given coin has scored."""
        value = self.board[column, height]
        if height >= 3:
            lower = height - 3
            col = self.board[column, lower:height]
            if (col == value).all():
                return True

        lower = max(0, column - 3)
        higher = min(7, column + 4)
        line = self.board[lower:higher, height]
        sublines = sliding_window_view(line, 4)
        fours = (sublines == value).all(axis=1)
        # print(line, sublines, fours)
        if fours.any():
            return True

        diag1 = diagonal(self.board, offset=column - height)
        if diag1.size >= 4:
            sublines = sliding_window_view(diag1, 4)
            fours = (sublines == value).all(axis=1)
            # print(diag1, sublines, fours)
            if fours.any():
                return True

        diag2 = diagonal(self.board, offset=column - height, axis1=1, axis2=0)
        if diag2.size >= 4:
            sublines = sliding_window_view(diag2, 4)
            fours = (sublines == value).all(axis=1)
            # print(diag2, sublines, fours)
            if fours.any():
                return True

        return False
