"""Connect Four game for mulitplayer Reinforcement Learning trial."""

from typing import TypedDict

from numpy import count_nonzero, diagonal, int_, ndarray, zeros, fliplr
from numpy.lib.stride_tricks import sliding_window_view


class MoveStatus(TypedDict):
    """Report the game status after `Game.step()`."""
    illegal: bool
    full: bool
    previous_player_won: bool
    diag: bool
    col: bool
    row: bool


class Game:
    """Connect Four game."""

    def __init__(self, height, width) -> None:
        # TODO(Jean): Variablize board size (Fanny's suggestion)
        self.height = height
        self.width = width
        self.board = zeros((width, height), dtype=int_)

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
            return MoveStatus(illegal=True, full=False, previous_player_won=False, diag=False, col=False, row=False)
        self.board[column, height] = player_id

        win = self.is_win(column, height)
        full = (self.board == 0).sum() == 0
        return MoveStatus(illegal=False, full=full, **win)

    def won_diag(self, value, column, height) -> bool:
        """Verify is the given coin has scored in diag."""
        value = self.board[column, height]

        diag1 = diagonal(self.board, offset=height - column)
        print(diag1)
        if diag1.size >= 4:
            sublines = sliding_window_view(diag1, 4)
            fours = (sublines == value).all(axis=1)
            if fours.any():
                return True

        diag2 = diagonal(fliplr(self.board),offset=(self.board.shape[1] - 1 - column) - height) 
        if diag2.size >= 4:
            sublines = sliding_window_view(diag2, 4)
            fours = (sublines == value).all(axis=1)
            if fours.any():
                return True
        return False

    def won_col(self, value, column, height) -> bool:
        """Verify is the given coin has scored in col."""
        value = self.board[column, height]
        if height >= 3:
            lower = height - 3
            col = self.board[column, lower:height]
            return (col == value).all()
        return False

    def won_row(self, value, column, height) -> bool:
        """Verify is the given coin has scored in row."""
        value = self.board[column, height]
        lower = max(0, column - 3)
        higher = min(self.width, column + 4)
        line = self.board[lower:higher, height]
        sublines = sliding_window_view(line, 4)
        fours = (sublines == value).all(axis=1)
        return fours.any()

    def is_win(self, column, height) -> bool:
        """Verify is the given coin has scored."""
        value = self.board[column, height]
        
        diag = self.won_diag(value, column, height)
        col = self.won_col(value, column, height)
        row = self.won_row(value, column, height)

        win = any([diag, col, row])    
        
        return {'previous_player_won': win, 'diag':diag, 'col':col, 'row':row} 
