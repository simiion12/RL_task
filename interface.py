import tempfile
import os
import matplotlib.pyplot as plt
from io import BytesIO
from unpickler import load_agent
import streamlit as st
from basic import TicTacToeEnv, GameRunner


# Set page config
st.set_page_config(
    page_title="Tic-Tac-Toe AI Challenge",
    page_icon="🎮",
    layout="wide"
)

def load_pretrained_agent():
    agent = load_agent("very_strong_agent.pkl")
    return agent

def render_board_as_image(board):
    """Render the board as a matplotlib figure"""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)

    # Draw grid
    for i in range(3):
        ax.axhline(i - 0.5, color='black', linewidth=2)
        ax.axvline(i - 0.5, color='black', linewidth=2)

    # Draw X's and O's
    for row in range(3):
        for col in range(3):
            if board[row, col] == 1:  # X
                ax.plot([col - 0.2, col + 0.2], [2 - row - 0.2, 2 - row + 0.2], 'r', linewidth=3)
                ax.plot([col - 0.2, col + 0.2], [2 - row + 0.2, 2 - row - 0.2], 'r', linewidth=3)
            elif board[row, col] == -1:  # O
                circle = plt.Circle((col, 2 - row), 0.2, fill=False, ec='blue', linewidth=3)
                ax.add_patch(circle)

    ax.set_xticks([])
    ax.set_yticks([])

    # Convert plot to image
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    return buf


def visualize_game(moves_history):
    """Create a visualization of the game using the moves history"""
    st.subheader("Game Visualization")

    # Create a TicTacToeEnv to track the board state
    env = TicTacToeEnv()
    board = env.reset()

    # Show initial empty board
    st.write("Starting position:")
    buf = render_board_as_image(board)
    st.image(buf)

    # Show each move
    for i, (player, move) in enumerate(moves_history):
        st.write(f"Move {i + 1}: Player {'X' if player == 1 else 'O'} plays at position {move}")
        env.step(move)
        buf = render_board_as_image(env.board)
        st.image(buf)

    # Show game result
    if env.winner == 0:
        st.write("Game ended in a draw!")
    else:
        st.write(f"Player {'X' if env.winner == 1 else 'O'} wins!")


def run_evaluation(uploaded_agent, pretrained_agent, num_games=100, visualize=False):
    """Run evaluation between uploaded agent and pretrained agent"""

    # Initialize agents
    agent1 = uploaded_agent  # Participant's agent (Player X)
    agent2 = pretrained_agent  # Pre-trained agent (Player O)

    game_runner = GameRunner(agent1, agent2)

    # If visualization is requested, run a single game and record moves
    if visualize:
        moves_history = []
        board = game_runner.env.reset()
        agent1.reset()
        agent2.reset()

        while True:
            # Determine current agent
            current_agent = agent1 if game_runner.env.current_player == 1 else agent2
            player_id = game_runner.env.current_player

            # Get action
            action = current_agent.get_action(board)
            moves_history.append((player_id, action))

            # Take action
            next_board, reward, done, info = game_runner.env.step(action)

            # Update board
            board = next_board

            if done:
                break

        # Visualize the recorded game
        visualize_game(moves_history)

        # Return result of this single game
        return {
            "agent1_wins": 1 if info["winner"] == 1 else 0,
            "agent2_wins": 1 if info["winner"] == -1 else 0,
            "draws": 1 if info["winner"] == 0 else 0,
            "visualized": True
        }

    # Run multiple games and collect stats
    else:
        with st.spinner(f"Running {num_games} games..."):
            progress_bar = st.progress(0)

            stats = {
                "agent1_wins": 0,
                "agent2_wins": 0,
                "draws": 0,
                "total_moves": 0
            }

            for i in range(num_games):
                # Alternate who goes first
                if i % 2 == 1:
                    game_runner.env.current_player = -1

                winner, num_moves = game_runner.run_game()
                stats["total_moves"] += num_moves

                if winner == 1:
                    stats["agent1_wins"] += 1
                elif winner == -1:
                    stats["agent2_wins"] += 1
                else:
                    stats["draws"] += 1

                # Update progress
                progress_bar.progress((i + 1) / num_games)

            stats["agent1_win_rate"] = stats["agent1_wins"] / num_games
            stats["agent2_win_rate"] = stats["agent2_wins"] / num_games
            stats["draw_rate"] = stats["draws"] / num_games
            stats["avg_moves_per_game"] = stats["total_moves"] / num_games

            return stats


def main():
    st.title("🎮 Tic-Tac-Toe AI Challenge")

    st.markdown("""
    ## Can your AI beat our pre-trained Tic-Tac-Toe agent?

    Welcome to the Tic-Tac-Toe AI Challenge! Your task is to develop an AI agent that can consistently 
    defeat our pre-trained AI in the classic game of Tic-Tac-Toe.

    ### Instructions:
    1. Upload your agent file (must be a pickled Python object implementing the BaseAgent interface)
    2. Choose the number of evaluation games to run
    3. Click "Run Evaluation" to test your agent against our pre-trained AI
    4. View the results and see if your agent meets the winning criteria

    If your agent achieves a win rate of at least 60%, you'll receive the flag!
    """)

    # Load pre-trained agent
    try:
        pretrained_agent = load_pretrained_agent()
        st.success("Pre-trained agent loaded successfully!")
    except Exception as e:
        st.error(f"Error loading pre-trained agent: {e}")
        return

    # File uploader for participant's agent
    uploaded_file = st.file_uploader("Upload your agent (pickle file)", type=["pkl"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_filepath = tmp_file.name

        try:
            # Load the agent
            uploaded_agent = load_agent(tmp_filepath)
            st.success("Your agent loaded successfully!")

            # Remove the temporary file
            os.unlink(tmp_filepath)

            # Evaluation options
            col1, col2 = st.columns(2)
            with col1:
                num_games = st.slider("Number of evaluation games", min_value=10, max_value=1000, value=100, step=10)
            with col2:
                visualize = st.checkbox("Visualize a sample game")

            # Run button
            if st.button("Run Evaluation"):
                results = run_evaluation(uploaded_agent, pretrained_agent, num_games, visualize)

                if not visualize:
                    # Display results
                    st.subheader("Evaluation Results")

                    # Create metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Your Win Rate", f"{results['agent1_win_rate']:.1%}")
                    with col2:
                        st.metric("Draw Rate", f"{results['draw_rate']:.1%}")
                    with col3:
                        st.metric("Opponent Win Rate", f"{results['agent2_win_rate']:.1%}")

                    # Display detailed stats
                    st.write("Detailed Statistics:")
                    st.write(f"- Total games: {num_games}")
                    st.write(f"- Your wins: {results['agent1_wins']}")
                    st.write(f"- Opponent wins: {results['agent2_wins']}")
                    st.write(f"- Draws: {results['draws']}")
                    st.write(f"- Average moves per game: {results['avg_moves_per_game']:.1f}")

                    # Check if participant passed the challenge
                    if results['agent1_win_rate'] >= 0.6:
                        st.balloons()
                        st.success("🎉 Congratulations! Your agent has defeated our AI!")
                        st.code("SIGMOID_MINIMAX", language="text")
                    else:
                        st.error("Your agent did not meet the winning criteria. Keep improving!")
                        st.write("Hint: Try to analyze the patterns in your opponent's strategy.")

        except Exception as e:
            st.error(f"Error evaluating your agent: {e}")

    # Documentation section
    with st.expander("Agent Implementation Guide"):
        st.markdown("""
        ## How to Implement Your Agent

        Your agent should inherit from the `BaseAgent` class and implement at least the `get_action` method:

        ```python
        class MyAgent(BaseAgent):
            def __init__(self, player_id):
                super().__init__(player_id)
                # Initialize your agent here

            def get_action(self, board):
                # Your move selection logic goes here
                # board is a 3x3 numpy array (0=empty, 1=X, -1=O)
                # Return a tuple of (row, col)
                # ...
        ```

        ### Tips:
        - You can download the example agent implementation from the sidebar
        - Consider using techniques like Minimax with alpha-beta pruning
        - Our agent has specific weaknesses that can be exploited!
        - Try to analyze your opponent's play patterns

        ### Saving Your Agent:
        ```python
        import pickle

        # Save your agent
        with open('my_agent.pkl', 'wb') as f:
            pickle.dump(my_agent, f)
        ```
        """)


if __name__ == "__main__":
    main()