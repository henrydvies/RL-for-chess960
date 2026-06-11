"""
MaskablePPO with MCTS action selection during rollout collection.
"""
import numpy as np
import torch as th
from sb3_contrib import MaskablePPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer, MaskableRolloutBuffer
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported

from engines.mcts.search import mcts_search
from game.board_representation import board_to_tensor
from utils.action_masks import action_masks as action_masks_helper


def board_policy_value(policy, device, board):
    """
    One forward pass: masked priors over legal moves and value for the side to move.
    """
    tensor_board = board_to_tensor(board)
    masks = action_masks_helper(board)

    obs = th.as_tensor(tensor_board.copy()).unsqueeze(0).float().to(device)
    mask = th.as_tensor(masks).unsqueeze(0).to(device)

    with th.no_grad():
        value = policy.predict_values(obs)
        dist = policy.get_distribution(obs, action_masks=mask)
        probs = dist.distribution.probs[0].cpu().numpy()

    legal = np.flatnonzero(masks)
    legal_probs = probs[legal]
    legal_probs = legal_probs / legal_probs.sum()
    priors = {int(action): float(prior) for action, prior in zip(legal, legal_probs)}
    return priors, float(value[0, 0].cpu())


def _boards_from_env(env):
    """
    Fetch a board copy from each parallel environment.
    """
    if not hasattr(env, "env_method"):
        raise ValueError("MCTS rollouts require a VecEnv that supports env_method")
    return env.env_method("get_board")


class MaskablePPO_MCTS(MaskablePPO):
    """
    MaskablePPO that selects learner actions with MCTS during rollout collection.
    """

    def __init__(
        self,
        policy,
        env,
        mcts_sims=0,
        mcts_c_puct=1.25,
        mcts_root_deterministic=False,
        **kwargs,
    ):
        self.mcts_sims = mcts_sims
        self.mcts_c_puct = mcts_c_puct
        self.mcts_root_deterministic = mcts_root_deterministic
        super().__init__(policy, env, **kwargs)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        use_masking: bool = True,
    ) -> bool:
        """
        Collect rollouts using MCTS action selection.
        """
        if self.mcts_sims <= 0:
            return super().collect_rollouts(
                env, callback, rollout_buffer, n_rollout_steps, use_masking
            )

        assert isinstance(
            rollout_buffer, (MaskableRolloutBuffer, MaskableDictRolloutBuffer)
        ), "RolloutBuffer doesn't support action masking"
        assert self._last_obs is not None, "No previous observation was provided"

        self.policy.set_training_mode(False)
        n_steps = 0
        action_masks = None
        rollout_buffer.reset()

        if use_masking and not is_masking_supported(env):
            raise ValueError(
                "Environment does not support action masking. Consider using ActionMasker wrapper"
            )

        callback.on_rollout_start()
        policy_value_fn = lambda board: board_policy_value(self.policy, self.device, board)
        rng = np.random.default_rng(self.seed)

        # Collect rollouts 
        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)

                if use_masking:
                    action_masks = get_action_masks(env)

                values = self.policy.predict_values(obs_tensor)
                boards = _boards_from_env(env)
                mcts_actions = []
                # Run MCTS for each board
                for board in boards:
                    action = mcts_search(
                        board,
                        policy_value_fn,
                        self.mcts_sims,
                        c_puct=self.mcts_c_puct,
                        root_deterministic=self.mcts_root_deterministic,
                        rng=rng,
                    )
                    mcts_actions.append(action)

                # Flatten the actions
                actions = np.array(mcts_actions).reshape(-1, 1)
                actions_tensor = th.as_tensor(actions).long().flatten().to(self.device)
                _, log_probs, _ = self.policy.evaluate_actions(
                    obs_tensor,
                    actions_tensor,
                    action_masks=action_masks,
                )

            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs

            # update locals and callback
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            # update rewards for terminal states
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            # Add the rollout to the buffer
            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                action_masks=action_masks,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True
