"""
Tests for MaskablePPO_MCTS in engines/rl/maskable_ppo_mcts.py
"""
from unittest.mock import patch

import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from engines.random.random_agent import RandomAgent
from engines.rl.policy_network import ChessPolicy, PolicyNetwork
from engines.rl.maskable_ppo_mcts import MaskablePPO_MCTS
from game.environment import ChessEnvironment


def _make_vec_env():
    """
    Single-process vector env for fast rollout smoke tests
    """
    return DummyVecEnv([lambda: ChessEnvironment(RandomAgent())])


def _policy_kwargs():
    return dict(
        features_extractor_class=PolicyNetwork,
        net_arch=[],
    )


## Testing rollout collection

def test_mcts_ppo_short_rollout():
    """
    A short learn() call with MCTS action selection should complete without error
    """
    env = _make_vec_env()
    model = MaskablePPO_MCTS(
        ChessPolicy,
        env,
        n_steps=4,
        mcts_sims=2,
        policy_kwargs=_policy_kwargs(),
        verbose=0,
        device="cpu",
    )
    model.learn(total_timesteps=4)
    env.close()


def test_mcts_sims_zero_uses_default_rollouts():
    """
    mcts_sims=0 should delegate to standard MaskablePPO collect_rollouts
    """
    env = _make_vec_env()
    model = MaskablePPO_MCTS(
        ChessPolicy,
        env,
        n_steps=4,
        mcts_sims=0,
        policy_kwargs=_policy_kwargs(),
        verbose=0,
        device="cpu",
    )
    with patch("engines.rl.maskable_ppo_mcts.mcts_search") as mock_search:
        model.learn(total_timesteps=4)
        mock_search.assert_not_called()
    env.close()


def test_mcts_rollout_stores_log_probs():
    """
    MCTS-chosen actions should still produce stored log probabilities
    """
    env = _make_vec_env()
    model = MaskablePPO_MCTS(
        ChessPolicy,
        env,
        n_steps=4,
        mcts_sims=2,
        policy_kwargs=_policy_kwargs(),
        verbose=0,
        device="cpu",
    )
    model.learn(total_timesteps=4)
    assert model.rollout_buffer.log_probs.shape[0] == 4
    assert model.rollout_buffer.actions.shape[0] == 4
    env.close()
