import numpy as np
from typing import Tuple
import random


class BaseAgent:
    """
    Base class for Tic-Tac-Toe agents.
    Participants should implement this interface.
    """

    def __init__(self, player_id: int):
        """
        Initialize the agent.

        Args:
            player_id: 1 for player X, -1 for player O
        """
        self.player_id = player_id

    def get_action(self, board: np.ndarray) -> Tuple[int, int]:
        """
        Select an action based on the current board state.

        Args:
            board: 3x3 numpy array representing the board state
                  (0 = empty, 1 = X, -1 = O)

        Returns:
            Tuple of (row, col) coordinates for the selected move
        """
        raise NotImplementedError("Agents must implement get_action method")

    def reset(self):
        """
        Reset the agent's state at the beginning of a new game.
        """
        pass

    def update(self, board: np.ndarray, action: Tuple[int, int], reward: float, next_board: np.ndarray, done: bool):
        """
        Update the agent based on experience (useful for learning agents).

        Args:
            board: Board state before action
            action: Action taken (row, col)
            reward: Reward received
            next_board: Board state after action
            done: Whether the game is over
        """
        pass

class RandomAgent(BaseAgent):
    """
    A simple agent that selects random valid moves.
    Provided as an example implementation.
    """

    def get_action(self, board: np.ndarray) -> Tuple[int, int]:
        valid_moves = []
        for row in range(3):
            for col in range(3):
                if board[row, col] == 0:
                    valid_moves.append((row, col))

        if not valid_moves:
            # No valid moves, should not happen if called correctly
            raise ValueError("No valid moves available")

        return random.choice(valid_moves)

