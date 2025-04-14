# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math
import os
import gdown


COLOR_MAP = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
    16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
    256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e",
    4096: "#3c3a32", 8192: "#3c3a32", 16384: "#3c3a32", 32768: "#3c3a32"
}
TEXT_COLOR = {
    2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2",
    32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2", 256: "#f9f6f2",
    512: "#f9f6f2", 1024: "#f9f6f2", 2048: "#f9f6f2", 4096: "#f9f6f2"
}

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False

        # Check vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action, return_after_state=False):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved  # Record if the move was valid

        if return_after_state:
            after_state = self.board.copy()
        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        if return_after_state:
            return self.board, self.score, done, after_state, {}
        return self.board, self.score, done, {}

    def render(self, mode="human", action=None):
        """
        Render the current board using Matplotlib.
        This function does not check if the action is valid and only displays the current board state.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")  # Default dark color
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.board, temp_board)

NUM_TILES = 16

def rotate_90(board_size, coords):
    """
    rotate 90 degrees counterclockwise
    """
    ret = []
    for coord in coords:
        ret.append((board_size-1 - coord[1], coord[0]))
    return ret

def reflect(board_size, coords):
    """
    reflect over horizontal axis
    """
    ret = []
    for coord in coords:
        ret.append((board_size-1 - coord[0], coord[1]))
    return ret

def log2(x):
    ans = 0
    while x > 1:
        x = x >> 1
        ans += 1

    if ans >= NUM_TILES:
        ans = NUM_TILES - 1

    return ans

class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        self.board_size = board_size
        self.patterns = patterns
        # Create a weight dictionary for each pattern (shared within a pattern group)
        self.weights = []
        # Generate symmetrical transformations for each pattern
        self.symmetry_patterns = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            self.symmetry_patterns.extend(syms)

    def generate_symmetries(self, pattern):
        # TODO: Generate 8 symmetrical transformations of the given pattern.
        ret = []
        for i in range(4):
            ret.append(pattern)
            pattern = rotate_90(self.board_size, pattern)

        pattern = reflect(self.board_size, pattern)

        for i in range(4):
            ret.append(pattern)
            pattern = rotate_90(self.board_size, pattern)

        # transform list[ list[tuple[row, col]] ] to list[ tuple[list[row], list[col]] ]
        for i in range(len(ret)):
            coords = ret[i]
            row_indices = [coord[0] for coord in coords]
            col_indices = [coord[1] for coord in coords]
            ret[i] = (row_indices, col_indices)

        return ret

    def get_feature(self, board, coords):
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        index = 0
        c = 1
        for tile in board[coords]:
            index += log2(tile) * c
            c *= NUM_TILES
        return index

    def value(self, board):
        # TODO: Estimate the board value: sum the evaluations from all patterns.
        value = 0
        for i, coords in enumerate(self.symmetry_patterns):
            feature = self.get_feature(board, coords)
            value += self.weights[i // 8][feature]
        return value

    def update(self, board, delta, alpha):
        # TODO: Update weights based on the TD error.
        for i, coords in enumerate(self.symmetry_patterns):
            feature = self.get_feature(board, coords)
            self.weights[i // 8][feature] += alpha * delta / len(self.symmetry_patterns)

class MCTS_Node:
    def __init__(self, state, is_after_state, score, env, parent, action):
        """
        state: current board state (numpy array)
        score: cumulative score at this node starting from the tree root
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.state = state
        self.is_after_state = is_after_state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = []
        self.prob = []  # for after states
        self.visits = 0
        self.total_reward = 0.0
        # List of untried actions based on the current state's legal moves
        self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]

    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0

class MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41,
                 gamma=0.99, max_branches=float("inf")):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.gamma = gamma
        self.max_branches = max_branches

    def create_env_from_state(self, state, score):
        # Create a deep copy of the environment with the given state and score.
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node: MCTS_Node):
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent.visits)/child.visits) to select the best child.
        if node.is_after_state:
            return np.random.choice(node.children, p=node.prob)
        else:
            # exploits = [child.total_reward / child.visits for child in node.children]
            # max_exploit = max(exploits)
            # min_exploit = min(exploits)
            # exploit_range = max_exploit - min_exploit

            best_child = None
            highest_ucb = float("-inf")
            for child in node.children:
                # normalize the exploit term to -1 ~ 1
                # if exploit_range != 0:
                #     exploit = (child.total_reward/child.visits - min_exploit) / exploit_range * 2 - 1
                # else:
                #     exploit = child.total_reward/child.visits

                exploit = child.total_reward / child.visits

                # exploit = child.total_reward / child.visits / 200

                ucb = exploit + self.c * np.sqrt(np.log(node.visits) / child.visits)
                if ucb > highest_ucb:
                    best_child = child
                    highest_ucb = ucb

            return best_child

    def rollout(self, node: MCTS_Node):
        if node.is_after_state:
            return node.total_reward
        else:
            if len(node.children) == 0:  # terminal state
                return 0

            highest_value = float("-inf")
            for child in node.children:
                # value = child.score + self.approximator.value(child.state)
                value = (child.score + self.approximator.value(child.state) * 5) / 800
                child.total_reward = value
                child.visits = 1

                if value > highest_value:
                    highest_value = value

            return highest_value

    def backpropagate(self, node, reward):
        # TODO: Propagate the obtained reward back up the tree.
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def run_simulation(self, root: MCTS_Node):
        node = root

        # TODO: Selection: Traverse the tree until reaching an unexpanded node.
        while len(node.children) > 0:
            node = self.select_child(node)

        # TODO: Expansion: If the node is not terminal, expand an untried action.
        if node.is_after_state:
            new_tiles = []
            for i in range(4):
                for j in range(4):
                    if node.state[i, j] == 0:
                        new_tiles.append((i, j, 2))
                        new_tiles.append((i, j, 4))

            if len(new_tiles) > self.max_branches:
                prob = np.array([9.0 if tile[2] == 2 else 1.0 for tile in new_tiles])
                prob /= prob.sum()
                indices = np.random.choice(len(new_tiles), self.max_branches, False, prob)
                new_tiles = np.array(new_tiles)[indices]

            for r, c, tile in new_tiles:
                child_state = node.state.copy()
                child_state[r, c] = tile
                sim_env = self.create_env_from_state(child_state, node.score)

                child = MCTS_Node(child_state, False, node.score, sim_env, node, None)
                node.children.append(child)
                node.prob.append(9 if tile == 2 else 1)

            node.prob = np.array(node.prob, np.float32)
            node.prob /= node.prob.sum()
        else:
            for action in node.untried_actions:
                sim_env = self.create_env_from_state(node.state, node.score)
                _, reward, _, after_state, _ = sim_env.step(action, True)
                child = MCTS_Node(after_state, True, reward, sim_env, node, action)
                node.children.append(child)

        rollout_reward = self.rollout(node)
        # Backpropagate the obtained reward.
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        # Compute the normalized visit count distribution for each child of the root.
        total_visits = sum(child.visits for child in root.children)
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for child in root.children:
            distribution[child.action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = child.action
        return best_action, distribution


patterns = [[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
            [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (3, 1)],
            [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],
            [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (2, 2)],
            [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (2, 2)],
            [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 2)],
            [(0, 0), (0, 1), (1, 1), (2, 0), (2, 1), (3, 1)],
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 2)]]

weights_file = "approximator_weights.pkl"
if not os.path.isfile(weights_file):
    gdown.download(
        "https://drive.google.com/file/d/1ZkOky4BKYB12p12CdsdyAm72cZtn_1Za/view?usp=sharing",
        weights_file, fuzzy=True
    )

approximator = NTupleApproximator(4, patterns)
with open(weights_file, "rb") as file:
    approximator.weights = pickle.load(file)

def get_action(state, score):
    env = Game2048Env()
    env.board = state
    env.score = score

    mcts = MCTS(env, approximator, 100, 1.41, 0.99, 4)
    root = MCTS_Node(state, False, 0, env, None, None)  # Initialize the root node for MCTS

    # Run multiple simulations to construct and refine the search tree
    for _ in range(mcts.iterations):
        mcts.run_simulation(root)

    # Select the best action based on the visit distribution of the root's children
    best_action, visit_distribution = mcts.best_action_distribution(root)

    # return best_action, visit_distribution
    return best_action
