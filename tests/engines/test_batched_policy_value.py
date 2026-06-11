"""
Tests for batched policy/value inference used by parallel MCTS rollouts.
"""
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import chess
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from engines.random.random_agent import RandomAgent
from engines.rl.batched_policy_value import (
    BatchedPolicyValueFn,
    board_policy_value,
    board_policy_value_batch,
    parallel_mcts_actions,
)
from engines.rl.maskable_ppo_mcts import MaskablePPO_MCTS
from engines.rl.policy_network import ChessPolicy, PolicyNetwork
from game.environment import ChessEnvironment


@pytest.fixture
def real_policy():
    env = DummyVecEnv([lambda: ChessEnvironment(RandomAgent())])
    model = MaskablePPO_MCTS(
        ChessPolicy,
        env,
        n_steps=4,
        mcts_sims=0,
        policy_kwargs=dict(features_extractor_class=PolicyNetwork, net_arch=[]),
        verbose=0,
        device="cpu",
    )
    model.policy.set_training_mode(False)
    yield model.policy, model.device
    env.close()


def test_board_policy_value_batch_matches_single(real_policy):
    """
    Batched inference should match single-board results.
    """
    policy, device = real_policy
    boards = [chess.Board(), chess.Board.from_chess960_pos(123)]

    single = [board_policy_value(policy, device, board) for board in boards]
    batched = board_policy_value_batch(policy, device, boards)

    assert len(batched) == 2
    for (priors_a, value_a), (priors_b, value_b) in zip(single, batched):
        assert set(priors_a) == set(priors_b)
        for action in priors_a:
            assert priors_a[action] == pytest.approx(priors_b[action])
        # Values can differ slightly between batch-1 and batch-N forwards on CPU float32
        assert value_a == pytest.approx(value_b, rel=1e-5, abs=1e-4)


def test_batched_policy_value_fn_coalesces_requests():
    """
    Concurrent requests should be processed in one batched forward call when possible.
    """
    policy = MagicMock()
    call_count = {"n": 0}

    def fake_batch(policy_arg, device_arg, boards):
        call_count["n"] += 1
        return [({"0": 1.0}, 0.1) for _ in boards]

    batched_fn = BatchedPolicyValueFn(policy, "cpu", max_batch=4, max_wait_s=0.01)

    with patch(
        "engines.rl.batched_policy_value.board_policy_value_batch",
        side_effect=fake_batch,
    ):
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(batched_fn, chess.Board.from_chess960_pos(i))
                for i in range(4)
            ]
            results = [future.result() for future in futures]

    assert len(results) == 4
    assert call_count["n"] == 1


def test_parallel_mcts_actions_returns_one_action_per_board():
    """
    Parallel rollout search should return an action for every env board.
    """
    boards = [chess.Board(), chess.Board.from_chess960_pos(42)]
    policy = MagicMock()

    with patch("engines.rl.batched_policy_value.mcts_search", side_effect=[7, 13]) as mock_search:
        actions = parallel_mcts_actions(
            boards,
            policy,
            "cpu",
            n_sims=2,
            c_puct=1.25,
            root_deterministic=True,
            base_seed=0,
        )

    assert actions == [7, 13]
    assert mock_search.call_count == 2


def test_parallel_mcts_actions_single_board_skips_thread_pool():
    """
    A single env should still work without thread overhead.
    """
    with patch("engines.rl.batched_policy_value.mcts_search", return_value=5) as mock_search, patch(
        "engines.rl.batched_policy_value.ThreadPoolExecutor"
    ) as mock_pool:
        actions = parallel_mcts_actions(
            [chess.Board()],
            MagicMock(),
            "cpu",
            n_sims=2,
            c_puct=1.25,
            root_deterministic=True,
            base_seed=0,
        )

    assert actions == [5]
    mock_pool.assert_not_called()
    mock_search.assert_called_once()
