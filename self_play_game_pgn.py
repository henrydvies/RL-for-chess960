"""
Play a single game between two agents and export PGN for replay on chess websites.
Usage: python -m engines.rl.play_game
Paste the output PGN into https://lichess.org/paste to replay.
"""
import chess
import chess.pgn
from engines.rl.rl_agent import rlAgent
from engines.random.random_agent import RandomAgent
from engines.minimax.minimax_agent import MinimaxAgent
from game.environment import ChessEnvironment


def play_game(white_agent, black_agent, chess960=True):
    """
    Play a single game between two agents and return the PGN.
    """
    if chess960:
        board = chess.Board.from_chess960_pos(__import__('random').randint(0, 959))
    else:
        board = chess.Board()
    
    # Set up PGN game
    game = chess.pgn.Game()
    game.headers["FEN"] = board.fen()
    game.headers["SetUp"] = "1"
    game.headers["White"] = white_agent.__class__.__name__
    game.headers["Black"] = black_agent.__class__.__name__
    #game.headers["Variant"] = "Chess960" if chess960 else "Standard"
    node = game

    move_count = 0

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            action = white_agent.take_turn(board)
        else:
            action = black_agent.take_turn(board)

        from_square = int(action) // 64
        to_square = int(action) % 64
        move = chess.Move(from_square, to_square)

        if move not in board.legal_moves:
            print(f"Illegal move attempted — game ended early.")
            break

        node = node.add_variation(move)
        board.push(move)
        move_count += 1

    # Add result
    outcome = board.outcome()
    if outcome:
        result = outcome.result()
    else:
        result = "*"
    game.headers["Result"] = result

    print(f"\nGame finished in {move_count} moves. Result: {result}")
    print(f"\n--- PGN --- (paste into https://lichess.org/paste)\n")
    print(game)
    return str(game)


if __name__ == "__main__":
    # Load trained RL agent as white
    temp_env = ChessEnvironment(opponent=RandomAgent())
    rl = rlAgent(temp_env)
    rl.load("models/rl_agent")

    # Play against random agent
    play_game(white_agent=rl, black_agent=RandomAgent(), chess960=False)