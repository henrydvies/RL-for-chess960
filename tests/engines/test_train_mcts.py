"""
Tests for MCTS self-play wiring in engines/rl/train.py
"""
import chess

from engines.random.random_agent import RandomAgent
from engines.rl.train import MCTSOpponentWrapper, _make_env_fn, MCTS_SIMS_TRAIN
from game.environment import ChessEnvironment


def test_mcts_opponent_wrapper_calls_search():
    """
    MCTSOpponentWrapper should invoke take_turn with n_sims set
    """
    class StubAgent:
        def __init__(self):
            self.last_n_sims = None

        def take_turn(self, board, n_sims=0, c_puct=1.25, root_deterministic=True, rng=None):
            self.last_n_sims = n_sims
            return list(board.legal_moves)[0]

        def load(self, model_path):
            pass

    stub = StubAgent()
    wrapper = MCTSOpponentWrapper(stub, n_sims=10, root_deterministic=False)
    board = chess.Board()
    wrapper.take_turn(board)
    assert stub.last_n_sims == 10


def test_make_env_fn_self_play_use_mcts():
    """
    Self-play env factory should pass use_mcts through to ChessEnvironment
    """
    env_fn = _make_env_fn(RandomAgent, temperature=0.0, use_mcts=True)
    env = env_fn()
    assert env.use_mcts is True
    assert env.temperature == 0.0


def test_mcts_opponent_wrapper_load_delegates(tmp_path):
    """
    reload_opponent in the env should reach the wrapped agent via load()
    """
    class StubAgent:
        def __init__(self):
            self.loaded_path = None

        def take_turn(self, board, n_sims=0, c_puct=1.25, root_deterministic=True, rng=None):
            return list(board.legal_moves)[0]

        def load(self, model_path):
            self.loaded_path = model_path

    stub = StubAgent()
    wrapper = MCTSOpponentWrapper(stub, n_sims=MCTS_SIMS_TRAIN)
    path = str(tmp_path / "model")
    wrapper.load(path)
    assert stub.loaded_path == path
