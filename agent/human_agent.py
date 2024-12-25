"""Connect Four human player interface to face a RL agent."""


class HumanAgent:
    """Connect Four human player interface."""

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

    def render(self, board) -> None:
        """Display board to the human."""
        print(board)

    def result(self, new_obs, reward, done, info) -> None:
        """Display RL agent's `step()` output to the human."""
        print(new_obs, reward, done, info)
