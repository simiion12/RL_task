import numpy as np
import sys
from unpickler import load_agent

def render_board(board):
    symbols = {0: " ", 1: "X", -1: "O"}
    print("  0 1 2")
    for row in range(3):
        print(f"{row} ", end="")
        for col in range(3):
            print(symbols[board[row, col]], end="")
            if col < 2:
                print("|", end="")
        print()
        if row < 2:
            print("  -+-+-")
    print()


def get_valid_moves(board):
    valid_moves = []
    for row in range(3):
        for col in range(3):
            if board[row, col] == 0:
                valid_moves.append((row, col))
    return valid_moves


def check_game_end(board):
    for player in [1, -1]:
        for row in range(3):
            if np.all(board[row, :] == player):
                return True, player

        for col in range(3):
            if np.all(board[:, col] == player):
                return True, player

        if np.all(np.diag(board) == player) or np.all(np.diag(np.fliplr(board)) == player):
            return True, player

    if np.all(board != 0):
        return True, 0

    return False, None


def play_against_agent(agent_path, human_player=-1):
    try:
        agent = load_agent(agent_path)
        print(f"Successfully loaded agent from {agent_path}")
    except Exception as e:
        print(f"Error loading agent: {e}")
        return

    agent.player_id = -human_player
    board = np.zeros((3, 3), dtype=np.int8)
    current_player = 1
    game_over = False
    winner = None

    print("\nGame started!")
    print("You are playing as", "X" if human_player == 1 else "O")
    print("Enter moves as 'row,col' (e.g., '1,1' for center)")
    print("Type 'quit' to exit\n")

    while not game_over:
        render_board(board)

        if current_player == human_player:
            valid_moves = get_valid_moves(board)
            if not valid_moves:
                game_over = True
                continue

            valid_input = False
            while not valid_input:
                move_str = input(f"Your move ({('X' if human_player == 1 else 'O')}): ")

                if move_str.lower() == 'quit':
                    print("Quitting game...")
                    return

                try:
                    row, col = map(int, move_str.strip().split(','))
                    move = (row, col)

                    if move in valid_moves:
                        valid_input = True
                    else:
                        print("Invalid move! Try again.")
                except ValueError:
                    print("Invalid input! Enter as 'row,col' (e.g., '1,1')")

            board[move] = human_player

        else:
            print(f"Agent's turn ({('X' if agent.player_id == 1 else 'O')})...")
            move = agent.get_action(board)
            print(f"Agent plays: {move[0]},{move[1]}")
            board[move] = agent.player_id

        game_over, winner = check_game_end(board)
        current_player = -current_player

    render_board(board)

    if winner == human_player:
        print("Congratulations! You won!")
    elif winner == agent.player_id:
        print("The agent won. Better luck next time!")
    else:
        print("It's a draw!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        agent_path = sys.argv[1]
    else:
        agent_path = input("Enter path to agent file (.pkl): ")

    player_choice = input("Do you want to play as X or O? (X goes first) [X/O]: ").upper()
    human_player = 1 if player_choice == 'X' else -1

    play_against_agent(agent_path, human_player)