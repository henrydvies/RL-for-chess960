"""
play N games between agents and update Elo ratings.
"""
import chess
from engines.random.random_agent import RandomAgent
from evaluation.elo_tracker import EloTracker


def play_single_game(white_agent, black_agent):
    """
    Play a single game between two agents.
    Returns True if white wins, False if black wins, None for draw.
    """
    board = chess.Board()

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            action = white_agent.take_turn(board)
        else:
            action = black_agent.take_turn(board)

        from_square = int(action) // 64
        to_square = int(action) % 64
        move = chess.Move(from_square, to_square)

        if move not in board.legal_moves:
            # Illegal move
            return board.turn != chess.WHITE

        board.push(move)

    outcome = board.outcome()
    if outcome is None:
        return None
    if outcome.winner == chess.WHITE:
        return True
    if outcome.winner == chess.BLACK:
        return False
    return None


def evaluate(rl_agent, opponent, n_games=20, tracker=None):
    """
    Play N games between rl_agent (white) and opponent (black).
    Updates Elo tracker in memory, saves once at the end.
    """
    if tracker is None:
        tracker = EloTracker()

    rl_name = "rlAgent"
    opponent_name = opponent.__class__.__name__

    # Ensure both agents are in the tracker
    if rl_name not in tracker.elo_map:
        tracker.elo_map[rl_name] = 600
    if opponent_name not in tracker.elo_map:
        tracker.elo_map[opponent_name] = 600

    wins, draws, losses = 0, 0, 0

    for _ in range(n_games):
        result = play_single_game(rl_agent, opponent)
        # Update Elo in memory
        tracker.update(rl_name, opponent_name, result)
        if result is True:
            wins += 1
        elif result is None:
            draws += 1
        else:
            losses += 1

    # Save once after all games
    tracker.save()
    return tracker