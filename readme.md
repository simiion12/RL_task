# Tic-Tac-Toe AI Challenge

## Overview
Welcome to the Tic-Tac-Toe AI Challenge! Your mission is to develop an AI agent that can consistently defeat our pre-trained Tic-Tac-Toe AI. This challenge is designed to test your skills in reinforcement learning and game strategy.

## Challenge Description
You need to create an AI agent that can play Tic-Tac-Toe at a high level, capable of winning against our pre-trained model with a consistent win rate.

## Requirements

1. Your agent **must** inherit from the `BaseAgent` class provided in `basic_agent.py`
2. Your agent **must** implement at least the `get_action(self, board)` method that takes the current board state and returns a valid move
3. Your solution should use reinforcement learning techniques (Q-learning, SARSA, Deep Q-Networks, etc.)
4. Your final submission should be a pickled agent (`.pkl` file)

## How Your Agent Will Be Evaluated

1. Your agent will play 100 games against our pre-trained agent
2. To unlock the flag, your agent must achieve at least a 60% win rate
3. Your agent will play both as the first player (X) and second player (O)
4. Games that end in a draw will not count toward your win percentage

## Implementation Guidelines

### Required Methods

```python
def get_action(self, board: np.ndarray) -> Tuple[int, int]:
    """
    Select an action based on the current board state.
    
    Args:
        board: 3x3 numpy array representing the board state
              (0 = empty, 1 = X, -1 = O)
              
    Returns:
        Tuple of (row, col) coordinates for the selected move
    """
    # Your implementation here
```

### Optional Methods (Helpful for Reinforcement Learning)

```python
def reset(self):
    """
    Reset the agent's state at the beginning of a new game.
    """
    # Reset any tracking variables
    
def update(self, board, action, reward, next_board, done):
    """
    Update the agent based on experience (useful for learning agents).
    """
    # Update Q-values or model weights
```

## Board Representation

- The board is represented as a 3x3 NumPy array
- Empty cells are represented by 0
- Player X (first player) is represented by 1
- Player O (second player) is represented by -1

## Training Tips

1. **Train against multiple opponents**: Don't just train against one strategy
2. **Explore different reward schemes**: Consider different reward signals beyond just win/loss
3. **Implement exploration strategies**: Epsilon-greedy, softmax, or UCB can help exploration
4. **Consider state representation**: How you represent the board state can affect learning efficiency
5. **Analyze our agent's weaknesses**: Try to identify patterns in how our pre-trained agent plays

## Saving Your Agent

```python
import pickle

# Save your agent
with open('my_agent.pkl', 'wb') as f:
    pickle.dump(my_agent, f)
```

## Testing Your Agent Locally

We provide a simple testing framework in `test_agent.py` that you can use to test your agent against various opponents before submission.

## Submission

Upload your pickled agent file to the submission portal. The evaluation system will automatically test your agent against our pre-trained agent and provide you with the results.

Good luck, and may the best AI win!
