# gridworld_template.py
# -------------------------------------------------------------
# 5×5 “Frozen Lake” grid world + Q‑learning skeleton.
# Students will fill in the sections marked # TODO.
# -------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

# ------------------------------------------------------------------
# 0️⃣  Global configuration (feel free to change later)
# ------------------------------------------------------------------
ROWS, COLS = 5, 5  # size of the grid
START = (0, 0)  # top‑left corner
GOAL = (4, 4)  # bottom‑right corner

# holes: positions that finish the episode with a big penalty.
HOLES = {(1, 2), (2, 3), (3, 1), (3, 3)}  # you can edit / add more

ACTIONS = {
    0: (-1, 0),  # up
    1: (1, 0),  # down
    2: (0, -1),  # left
    3: (0, 1),  # right
}
N_ACTIONS = len(ACTIONS)


# ------------------------------------------------------------------
# 1️⃣  Simple environment – students only edit the doc‑strings.
# ------------------------------------------------------------------
class GridWorld:
    """
    Represent the frozen lake as a 2‑D NumPy array.
    The state is stored as a (row, col) tuple.
    """

    def __init__(self):
        self.reset()

    # --------------------------------------------------------------
    def reset(self) -> Tuple[int, int]:
        """
        Put the agent back at the START position.
        Returns the initial state.
        """
        # TODO: set the current position to START and return it
        self.pos = START
        return self.pos

    # --------------------------------------------------------------
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        Apply *action* (0=up,1=down,2=left,3=right).

        Returns:
            next_state (row, col)
            reward (float)
            done   (bool) – True if we reached a hole or the goal.
        """
        # TODO: compute the new position respecting the grid borders.
        #       Use ACTIONS[action] to get the delta.
        dr, dc = ACTIONS[action]
        r, c = self.pos
        nr, nc = r + dr, c + dc

        # stay inside the board
        if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS:
            nr, nc = r, c  # illegal move → stay where we are

        self.pos = (nr, nc)

        # TODO: decide the reward and termination flag
        if self.pos == GOAL:
            return self.pos, 1.0, True  # reached the goal
        if self.pos in HOLES:
            return self.pos, -1.0, True  # fell in a hole
        return self.pos, -0.01, False  # normal step (small penalty)

    # --------------------------------------------------------------
    def get_valid_actions(self) -> List[int]:
        """
        In this simple grid all 4 actions are always legal
        (the step function will just keep you in place if you hit a wall).
        Returns the list [0,1,2,3].
        """
        # TODO: return a list of all action indices
        return list(ACTIONS.keys())

    # --------------------------------------------------------------
    def render(self) -> None:
        """
        Print a tiny ASCII picture of the board.
        S = start, G = goal, H = hole, A = agent.
        """
        # TODO: build a 2‑D char array and print it.
        board = np.full((ROWS, COLS), ".")
        for hr, hc in HOLES:
            board[hr, hc] = "H"
        board[GOAL] = "G"
        board[START] = "S"
        r, c = self.pos
        board[r, c] = "A"
        for row in board:
            print(" ".join(row))
        print()


# ------------------------------------------------------------------
# 2️⃣  Q‑Learning agent – students fill in the learning rule.
# ------------------------------------------------------------------
class QLearningAgent:
    """
    Tabular Q‑learning agent.  The Q‑table is a NumPy array
    with shape (n_states, n_actions).
    """

    def __init__(self, lr: float = 0.1, gamma: float = 0.99, eps: float = 0.2):
        self.lr = lr  # learning rate α
        self.gamma = gamma  # discount factor γ
        self.eps = eps  # ε for ε‑greedy exploration

        # TODO: initialise a Q‑table of zeros.
        #       Number of states = ROWS * COLS.
        self.Q = np.zeros((ROWS * COLS, N_ACTIONS))

    # --------------------------------------------------------------
    def _state_to_index(self, state: Tuple[int, int]) -> int:
        """Convert (row, col) into a single integer 0 … ROWS*COLS‑1."""
        r, c = state
        return r * COLS + c

    # --------------------------------------------------------------
    def select_action(self, state: Tuple[int, int]) -> int:
        """
        ε‑greedy policy.
        With probability eps pick a random valid action,
        otherwise pick the action with the highest Q‑value.
        """
        # TODO: implement the ε‑greedy choice.
        if np.random.rand() < self.eps:
            return np.random.choice(N_ACTIONS)
        idx = self._state_to_index(state)
        # break ties randomly
        max_q = np.max(self.Q[idx])
        best = np.where(self.Q[idx] == max_q)[0]
        return np.random.choice(best)

    # --------------------------------------------------------------
    def update(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int],
        done: bool,
    ) -> None:
        """
        Apply the Q‑learning update rule:
        Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') – Q(s,a) ]
        """
        # TODO: write the update formula.
        s_idx = self._state_to_index(state)
        ns_idx = self._state_to_index(next_state)
        target = reward
        if not done:
            target += self.gamma * np.max(self.Q[ns_idx])
        self.Q[s_idx, action] += self.lr * (target - self.Q[s_idx, action])


# ------------------------------------------------------------------
# 3️⃣  Training loop (still a skeleton)
# ------------------------------------------------------------------
def train_agent(
    num_episodes: int = 200, max_steps: int = 100, render_every: int = 0
) -> QLearningAgent:
    """
    Run the Q‑learning algorithm for *num_episodes*.
    If render_every > 0 the environment is printed every that many episodes.
    Returns the trained agent.
    """
    env = GridWorld()
    agent = QLearningAgent()

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        step = 0

        while not done and step < max_steps:
            # 1️⃣ Choose an action
            action = agent.select_action(state)

            # 2️⃣ Apply it to the environment
            next_state, reward, done = env.step(action)

            # 3️⃣ Update Q‑table
            agent.update(state, action, reward, next_state, done)

            state = next_state
            step += 1

        # OPTIONAL visualisation of the board every few episodes
        if render_every and ep % render_every == 0:
            print(f"\nEpisode {ep}")
            env.render()
            plot_value_function(agent, title=f"Episode {ep}")

    return agent


# ------------------------------------------------------------------
# 4️⃣  Helper to plot a heat‑map of the learned state‑value (max Q)
# ------------------------------------------------------------------
def plot_value_function(agent: QLearningAgent, title: str = "Value function"):
    """
    Show a 5×5 heat map where each cell contains max_a Q(s,a).
    """
    # TODO: reshape the max‑Q values into a ROWS×COLS matrix and plot.
    V = np.max(agent.Q, axis=1).reshape((ROWS, COLS))
    plt.figure(figsize=(5, 4))
    plt.title(title)
    plt.imshow(V, cmap="viridis", origin="upper")
    plt.colorbar(label="max Q")
    plt.xticks(np.arange(COLS))
    plt.yticks(np.arange(ROWS))
    plt.show()


# ------------------------------------------------------------------
# 5️⃣  Main entry point – students just run it.
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Feel free to change the numbers for extra practice.
    trained_agent = train_agent(
        num_episodes=300, max_steps=50, render_every=0
    )  # set to e.g. 50 to watch progress

    # Final visualisation
    plot_value_function(trained_agent, title="Final learned value function")
