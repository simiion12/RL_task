import numpy as np
from typing import Tuple, Dict, Any, List
import pickle
from basic_agent import BaseAgent


class TicTacToeEnv:
    """
    A Tic-Tac-Toe environment that follows a gym-like interface.
    """

    def __init__(self):
        # Board is represented as a 3x3 numpy array:
        # 0 = empty, 1 = X (player 1), -1 = O (player 2)
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.current_player = 1  # Player 1 (X) starts
        self.done = False
        self.winner = None

    def reset(self) -> np.ndarray:
        """Reset the game to initial state and return the board."""
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.board.copy()

    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute action and return new state, reward, done flag, and info.
        Action is a tuple of (row, col) coordinates where the current player wants to place their mark.

        Returns:
            observation: Board state after action
            reward: Reward received
            done: Whether the game is over
            info: Additional information
        """
        row, col = action

        # Check if the move is valid
        if self.board[row, col] != 0 or self.done:
            # Invalid move, return current state with negative reward
            return self.board.copy(), -10.0, self.done, {"error": "Invalid move", "winner": self.winner}

        # Place the mark on the board
        self.board[row, col] = self.current_player

        # Check for game end conditions
        self._check_game_end()

        # Calculate reward
        reward = self._calculate_reward()

        # Switch to the other player if the game is not done
        if not self.done:
            self.current_player *= -1

        return self.board.copy(), reward, self.done, {"winner": self.winner}

    def _check_game_end(self) -> None:
        """Check if the game has ended (win or draw)."""
        # Check rows, columns, and diagonals for a win
        for player in [1, -1]:
            # Check rows
            for row in range(3):
                if np.all(self.board[row, :] == player):
                    self.done = True
                    self.winner = player
                    return

            # Check columns
            for col in range(3):
                if np.all(self.board[:, col] == player):
                    self.done = True
                    self.winner = player
                    return

            # Check diagonals
            if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):
                self.done = True
                self.winner = player
                return

        # Check for draw (board full)
        if np.all(self.board != 0):
            self.done = True
            self.winner = 0  # No winner (draw)

    def _calculate_reward(self) -> float:
        """Calculate reward based on game state."""
        if not self.done:
            return 0.0

        if self.winner == self.current_player:
            return 1.0  # Win
        elif self.winner == 0:
            return 0.2  # Draw
        else:
            return -1.0  # Loss

    def render(self) -> str:
        """Render the board as a string."""
        symbols = {0: " ", 1: "X", -1: "O"}
        board_str = ""

        for row in range(3):
            for col in range(3):
                board_str += symbols[self.board[row, col]]
                if col < 2:
                    board_str += "|"
            if row < 2:
                board_str += "\n-+-+-\n"

        return board_str

    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """Return list of valid moves as (row, col) tuples."""
        if self.done:
            return []

        valid_moves = []
        for row in range(3):
            for col in range(3):
                if self.board[row, col] == 0:
                    valid_moves.append((row, col))

        return valid_moves


class GameRunner:
    """
    Utility class to run games between two agents.
    """

    def __init__(self, agent1: BaseAgent, agent2: BaseAgent):
        self.env = TicTacToeEnv()
        self.agent1 = agent1  # Player X (1)
        self.agent2 = agent2  # Player O (-1)

    def run_game(self, render: bool = False) -> Tuple[int, int]:
        """
        Run a single game between the two agents.

        Args:
            render: Whether to print the board after each move

        Returns:
            Tuple of (winner, num_moves)
            winner: 1 for agent1 win, -1 for agent2 win, 0 for draw
            num_moves: Number of moves in the game
        """
        board = self.env.reset()
        self.agent1.reset()
        self.agent2.reset()

        num_moves = 0

        while True:
            # Determine current agent
            current_agent = self.agent1 if self.env.current_player == 1 else self.agent2

            # Get action from current agent
            action = current_agent.get_action(board)

            # Take the action
            next_board, reward, done, info = self.env.step(action)

            # Update agent
            current_agent.update(board, action, reward, next_board, done)

            num_moves += 1

            if render:
                print(f"Player {'X' if self.env.current_player == 1 else 'O'} plays {action}")
                print(self.env.render())
                print()

            # Update board
            board = next_board

            if done:
                winner = info.get("winner", 0)
                if render:
                    if winner == 0:
                        print("Game ended in a draw")
                    else:
                        print(f"Player {'X' if winner == 1 else 'O'} wins!")

                return winner, num_moves

    def run_multiple_games(self, num_games: int = 100) -> Dict[str, Any]:
        """
        Run multiple games and return statistics.

        Args:
            num_games: Number of games to run

        Returns:
            Dictionary of statistics
        """
        stats = {
            "agent1_wins": 0,
            "agent2_wins": 0,
            "draws": 0,
            "total_moves": 0
        }

        for i in range(num_games):
            # Alternate who goes first
            if i % 2 == 1:
                self.env.current_player = -1

            winner, num_moves = self.run_game()
            stats["total_moves"] += num_moves

            if winner == 1:
                stats["agent1_wins"] += 1
            elif winner == -1:
                stats["agent2_wins"] += 1
            else:
                stats["draws"] += 1

        stats["agent1_win_rate"] = stats["agent1_wins"] / num_games
        stats["agent2_win_rate"] = stats["agent2_wins"] / num_games
        stats["draw_rate"] = stats["draws"] / num_games
        stats["avg_moves_per_game"] = stats["total_moves"] / num_games

        return stats


def save_agent(agent, filename):
    """
    Save an agent to a file using pickle.
    """
    with open(filename, 'wb') as f:
        pickle.dump(agent, f)
