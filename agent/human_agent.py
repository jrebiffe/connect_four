"""Connect Four human player interface to face a RL agent."""

from numpy import rot90

class HumanAgent:
    """Connect Four human player interface."""
    def __init__(self, env, agent_config) -> None:
        self.env = env

    def call_action(self) -> int:
        """Ask human the column in which to insert coin."""
        value: int | None = None
        while value is None:
            text = input("In which columns?")
            try:
                value = int(text)
            except ValueError as exception:
                print(f"'{text}' is not an int: {exception}")
            if value not in range(7):
                print(f"'{value}' is not in range(7)")
                value = None

        return value

    def observe(self, board) -> None:
        """Display board to the human."""
        board = rot90(board)
        print(" ", end="")
        for index, line in enumerate(board[0, :]):
            print(index, end=" ")
        print()

        for line in board:
            # print("|", end="")
            for column in line:
                print("|", end="")
                if column == 0:
                    print(" ", end="")
                elif column == 1:
                    print("\033[91mX\033[0m", end="")
                elif column == 2:
                    print("\033[93mO\033[0m", end="")
            print("|")

    def result(self, new_obs, reward, done, info) -> None:
        """Display RL agent's `step()` output to the human."""
        self.observe(new_obs)
        if info['win']:
            print('You won!')
        if info['loose']:
            print('Looooooser')
        # print(reward, done, info)
