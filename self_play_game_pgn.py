"""
Play a single game between two agents and export PGN for replay on chess websites.
Usage: python self_play_game_pgn.py
Paste the output PGN into https://lichess.org/paste to replay.
"""
import chess
import chess.pgn
from engines.rl.rl_agent import rlAgent
from engines.random.random_agent import RandomAgent
from engines.rl.train import MCTS_SIMS_TRAIN, MCTS_C_PUCT
from game.environment import ChessEnvironment


def play_game(white_agent, black_agent, chess960=True, n_sims=MCTS_SIMS_TRAIN, root_deterministic=True):
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
    node = game

    move_count = 0

    while not board.is_game_over() and move_count < 300:
        agent = white_agent if board.turn == chess.WHITE else black_agent
        if isinstance(agent, rlAgent) and n_sims > 0:
            move = agent.take_turn(
                board,
                n_sims=n_sims,
                c_puct=MCTS_C_PUCT,
                root_deterministic=root_deterministic,
            )
        else:
            move = agent.take_turn(board)

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

    if move_count >= 300:
        result = "1/2-1/2"
    game.headers["Result"] = result

    print(f"\nGame finished in {move_count} moves. Result: {result}")
    print(f"\n--- PGN --- (paste into https://lichess.org/paste)\n")
    print(game)
    return str(game)


if __name__ == "__main__":
    temp_env = ChessEnvironment(opponent=RandomAgent())
    rl = rlAgent(temp_env)
    rl.load("models/rl_agent_v4/rl_agent_v4")

    opp_agent = rlAgent(temp_env)
    opp_agent.load("models/rl_agent_v4/rl_agent_v4")

    play_game(
        white_agent=rl,
        black_agent=opp_agent,
        chess960=True,
        n_sims=MCTS_SIMS_TRAIN,
        root_deterministic=True,
    )