"""
PUCT MCTS
"""
import chess
import numpy as np

from utils.action_masks import action_masks, action_to_move


class _MCTSNode:
    """
    Single node in the search tree. Values are stored from the perspective of
    board.turn at this node.
    """

    def __init__(self, board, parent=None, action=None, prior=0.0):
        self.board = board.copy()
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.expanded = False

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def _terminal_value(board):
    """
    Outcome value from the side-to-move perspective. Win=+1, draw=0, loss=-1.
    """
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    if outcome.winner == board.turn:
        return 1.0
    return -1.0


def _select_child(node, c_puct):
    """
    Pick the child with highest PUCT score.
    """
    sqrt_parent = np.sqrt(node.visit_count)

    best_score = -np.inf
    best_action = None
    for action, child in node.children.items():
        u = c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
        # Child Q is from the side-to-move at the child, flip for the parent
        score = -child.q_value + u
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def _expand(node, get_policy_value):
    """
    Attach children for every legal move and mark the node expanded.
    Returns the leaf value used for backup (network or terminal).
    """
    terminal = _terminal_value(node.board)
    if terminal is not None:
        node.expanded = True
        return terminal

    priors, value = get_policy_value(node.board)
    for action, prior in priors.items():
        move = action_to_move(action, node.board)
        if move not in node.board.legal_moves:
            continue
        child_board = node.board.copy()
        child_board.push(move)
        node.children[action] = _MCTSNode(
            child_board,
            parent=node,
            action=action,
            prior=prior,
        )

    node.expanded = True
    return float(value)


def _backup(path, value):
    """
    Propagate leaf value up the tree, flipping sign at each ply.
    """
    for node in reversed(path):
        node.visit_count += 1
        node.value_sum += value
        value = -value


def mcts_search(
    board,
    get_policy_value,
    n_sims,
    c_puct=1.25,
    root_deterministic=False,
    rng=None,
):
    """
    Run MCTS from board and return the chosen action index.

    get_policy_value(board) should return (priors, value) where priors maps
    legal action indices to prior probabilities (summing to ~1 over legal
    moves) and value is a scalar from the side-to-move perspective.
    """
    if n_sims <= 0:
        raise ValueError("n_sims must be positive for mcts_search")

    if rng is None:
        rng = np.random.default_rng()

    root = _MCTSNode(board)

    for _ in range(n_sims):
        node = root
        path = [node]

        # Selection: descend through expanded nodes
        while node.expanded and node.children:
            action = _select_child(node, c_puct)
            node = node.children[action]
            path.append(node)

        # Expansion and leaf evaluation
        if not node.expanded:
            value = _expand(node, get_policy_value)
        else:
            # Terminal leaf visited again
            value = _terminal_value(node.board)
            if value is None:
                value = node.q_value

        _backup(path, value)

    if not root.children:
        # No legal moves
        mask = action_masks(board)
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            raise ValueError("MCTS root has no legal moves")
        return int(legal[0])

    visits = np.array([root.children[a].visit_count for a in root.children], dtype=float)
    actions = list(root.children.keys())

    if root_deterministic:
        return actions[int(np.argmax(visits))]

    visit_total = visits.sum()
    if visit_total <= 0:
        return actions[int(rng.integers(len(actions)))]
    probs = visits / visit_total
    return actions[int(rng.choice(len(actions), p=probs))]
