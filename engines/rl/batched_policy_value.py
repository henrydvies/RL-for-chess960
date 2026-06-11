"""
Thread-safe batched policy/value inference for parallel MCTS across vector envs.
"""
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch as th

from engines.mcts.search import mcts_search
from game.board_representation import board_to_tensor
from utils.action_masks import action_masks as action_masks_helper


def board_policy_value(policy, device, board):
    """
    One forward pass: masked priors over legal moves and value for the side to move.
    """
    return board_policy_value_batch(policy, device, [board])[0]


def board_policy_value_batch(policy, device, boards):
    """
    Batched forward pass for multiple boards. Returns list of (priors, value) pairs.
    """
    if not boards:
        return []

    tensors = []
    masks_list = []
    for board in boards:
        tensor_board = board_to_tensor(board)
        tensors.append(tensor_board.copy() if not tensor_board.flags.writeable else tensor_board)
        masks_list.append(action_masks_helper(board))

    obs = th.as_tensor(np.stack(tensors)).float().to(device)
    mask = th.as_tensor(np.stack(masks_list)).to(device)

    with th.no_grad():
        policy.set_training_mode(False)
        values = policy.predict_values(obs)
        dist = policy.get_distribution(obs, action_masks=mask)
        probs = dist.distribution.probs.cpu().numpy()

    if values.dim() == 1:
        values = values.unsqueeze(-1)

    results = []
    for i, masks in enumerate(masks_list):
        legal = np.flatnonzero(masks)
        legal_probs = probs[i, legal]
        legal_probs = legal_probs / legal_probs.sum()
        priors = {int(action): float(prior) for action, prior in zip(legal, legal_probs)}
        results.append((priors, float(values[i, 0].cpu())))

    return results


class _PolicyValueRequest:
    __slots__ = ("board", "priors", "value")

    def __init__(self, board):
        self.board = board
        self.priors = None
        self.value = None


class BatchedPolicyValueFn:
    """
    Callable get_policy_value(board) that coalesces concurrent requests into batches.
    """

    def __init__(self, policy, device, max_batch=8, max_wait_s=0.001):
        self.policy = policy
        self.device = device
        self.max_batch = max(1, max_batch)
        self.max_wait_s = max_wait_s
        self._cond = threading.Condition()
        self._pending = []

    def __call__(self, board):
        request = _PolicyValueRequest(board)
        with self._cond:
            self._pending.append(request)
            while request.priors is None:
                if len(self._pending) >= self.max_batch:
                    self._process_batch_locked()
                else:
                    self._cond.wait(timeout=self.max_wait_s)
                if request.priors is None and self._pending:
                    self._process_batch_locked()
        return request.priors, request.value

    def _process_batch_locked(self):
        if not self._pending:
            return

        batch = self._pending
        self._pending = []
        boards = [request.board for request in batch]
        self._cond.release()
        try:
            results = board_policy_value_batch(self.policy, self.device, boards)
        finally:
            self._cond.acquire()

        for request, (priors, value) in zip(batch, results):
            request.priors = priors
            request.value = value
        self._cond.notify_all()


def parallel_mcts_actions(
    boards,
    policy,
    device,
    n_sims,
    c_puct,
    root_deterministic,
    base_seed,
):
    """
    Run MCTS for each board, batching network inference across concurrent searches.
    """
    if not boards:
        return []

    batched_fn = BatchedPolicyValueFn(policy, device, max_batch=len(boards))

    def _search_one(board_index, board):
        rng = np.random.default_rng(base_seed + board_index)
        return mcts_search(
            board,
            batched_fn,
            n_sims,
            c_puct=c_puct,
            root_deterministic=root_deterministic,
            rng=rng,
        )

    if len(boards) == 1:
        return [_search_one(0, boards[0])]

    with ThreadPoolExecutor(max_workers=len(boards)) as executor:
        futures = [
            executor.submit(_search_one, index, board)
            for index, board in enumerate(boards)
        ]
        return [future.result() for future in futures]
