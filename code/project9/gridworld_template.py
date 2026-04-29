#!/usr/bin/env python3
# -------------------------------------------------------------
# gridworld_student.py
# -------------------------------------------------------------
# Student version of the 5×5 Frozen‑Lake Q‑learning example.
# Fill in every "TODO" section following the step‑by‑step comments.
# -------------------------------------------------------------

import time
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle

# ------------------------------------------------------------------
# 0️⃣  Configuration (you can change values if you want)
# ------------------------------------------------------------------
ROWS, COLS = 5, 5
START = (0, 0)
GOAL = (4, 4)
HOLES = {(1, 2), (2, 3), (3, 1), (3, 3)}  # set of hole coordinates

ACTIONS = {
    0: (-1, 0),  # up
    1: (1, 0),  # down
    2: (0, -1),  # left
    3: (0, 1),  # right
}
N_ACTIONS = len(ACTIONS)


# ==================================================================
# 1️⃣  Environment – GridWorld
# ==================================================================
class GridWorld:
    """Simple frozen‑lake environment."""

    def __init__(self):
        """Create a fresh environment – simply call reset()."""
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> Tuple[int, int]:
        """
        Place the agent back at START and return the current state.
        """
        # TODO: set the internal position to START and return it
        self.pos = START
        return self.pos
    

    # ------------------------------------------------------------------
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        Execute `action` (0‑3) and return a tuple:
            (next_state, reward, done)
        """
        # TODO:
        #   1. Look up the delta (dr, dc) in ACTIONS.
        #   2. Compute the candidate new position (nr, nc).
        #   3. If the new position would leave the grid, keep the old one.
        #   4. Update self.pos.
        #   5. Return (pos, reward, done) according to:
        #        * GOAL  → reward  1.0, done True
        #        * HOLE  → reward -1.0, done True
        #        * otherwise → reward -0.01, done False
        dr,dc = ACTIONS[action]
        r,c = self.pos
        nr,nc = r + dr, c + dc
        if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS:
            nr, nc = r, c
        self.pos = tuple(nr,nc)
        if self.pos == GOAL:
            return self.pos, 1.0, True
        elif self.pos in HOLES:
            return self.pos, -1.0, True
        return self.pos, -0.01, False
                


    # ------------------------------------------------------------------
    def get_valid_actions(self) -> List[int]:
        """
        Return a list of action IDs that are allowed in the current state.
        In this simple world all four moves are always allowed
        (the step() method will simply bounce off walls).
        """
        # TODO: return the list of keys from ACTIONS
        pass

    # ------------------------------------------------------------------
    def render(self) -> None:
        """
        Print a quick ASCII representation of the board.
        Useful for debugging or the console play‑through.
        """
        # TODO:
        #   * create a 2‑D array of "." strings
        #   * set HOLE cells to "H", the GOAL to "G", START to "S"
        #   * mark the agent's current location with "A"
        #   * print each row joined by spaces, then a blank line
        pass


# ==================================================================
# 2️⃣  Q‑Learning Agent (tabular)
# ==================================================================
class QLearningAgent:
    """Tabular Q‑learning with ε‑greedy exploration."""

    def __init__(self, lr: float = 0.1, gamma: float = 0.99, eps: float = 0.2):
        """
        Initialise learning rate, discount factor, exploration rate,
        and a zero‑filled Q‑table of shape (ROWS*COLS, N_ACTIONS).
        """
        # TODO: store lr, gamma, eps and allocate the Q‑table
        pass

    # ------------------------------------------------------------------
    # Helper: convert a (row, col) state into a flat index 0 … ROWS*COLS‑1
    # ------------------------------------------------------------------
    def _state_to_index(self, state: Tuple[int, int]) -> int:
        # TODO: return row * COLS + col
        pass

    # ------------------------------------------------------------------
    # ε‑greedy action selection
    # ------------------------------------------------------------------
    def select_action(self, state: Tuple[int, int]) -> int:
        """
        With probability eps choose a random action,
        otherwise choose the action(s) with maximal Q-value
        (break ties randomly).
        """
        # TODO:
        #   * draw a uniform random number
        #   * if < eps → return np.random.choice(N_ACTIONS)
        #   * otherwise look up the row in Q, find max value, get all actions
        #     that achieve this max, and randomly pick one of them.
        pass

    # ------------------------------------------------------------------
    # Standard Q‑learning update
    # ------------------------------------------------------------------
    def update(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int],
        done: bool,
    ) -> None:
        """
        Apply the Q‑learning formula:
            Q(s,a) ← Q(s,a) + lr * (target - Q(s,a))

        where target = r + γ * max_a' Q(s',a')  (unless `done`).
        """
        # TODO:
        #   * translate state and next_state to flat indices
        #   * compute the target (add discounted max‑Q if not done)
        #   * perform the incremental update on Q[s_idx, action]
        pass


# ==================================================================
# 3️⃣  Training loop
# ==================================================================
def train_agent(
    num_episodes: int = 300,
    max_steps: int = 50,
    render_every: int = 0,
) -> QLearningAgent:
    """
    Run `num_episodes` episodes of interaction with a fresh GridWorld.
    Returns the trained QLearningAgent.
    """
    # TODO:
    #   1. Create a GridWorld instance.
    #   2. Create a QLearningAgent (you may use the default hyper‑parameters).
    #   3. Loop over episodes (1 … num_episodes)
    #       a. reset the environment → state
    #       b. loop over steps (max_steps) until done
    #           i.   choose action with agent.select_action(state)
    #           ii.  apply env.step(action) → next_state, reward, done
    #           iii. agent.update(state, action, reward, next_state, done)
    #           iv.  set state = next_state
    #       c. (optional) every `render_every` episodes call env.render()
    #          and plot_value_function() to visualise progress.
    #   4. Return the trained agent.
    pass


# ==================================================================
# 4️⃣  Plotting helpers (you don't need to modify these)
# ==================================================================
def plot_value_function(agent: QLearningAgent, title: str = "Value function"):
    """Draw a heat‑map of max_a Q(s,a)."""
    V = np.max(agent.Q, axis=1).reshape((ROWS, COLS))

    plt.figure(figsize=(5, 4))
    plt.title(title)
    plt.imshow(V, cmap="viridis", origin="upper")
    plt.colorbar(label="max Q")
    plt.xticks(np.arange(COLS))
    plt.yticks(np.arange(ROWS))

    for r in range(ROWS):
        for c in range(COLS):
            plt.text(
                c,
                r,
                f"{V[r, c]:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )
    plt.show()


def plot_policy(agent: QLearningAgent, title: str = "Greedy policy"):
    """Overlay arrows showing the best action for each state."""
    V = np.max(agent.Q, axis=1).reshape((ROWS, COLS))
    best_action = np.argmax(agent.Q, axis=1).reshape((ROWS, COLS))

    # arrow vectors (dy, dx) for each action id
    action_vec = {
        0: (-0.4, 0),  # up
        1: (0.4, 0),  # down
        2: (0, -0.4),  # left
        3: (0, 0.4),  # right
    }

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(V, cmap="viridis", origin="upper")
    fig.colorbar(im, ax=ax, label="max Q")
    ax.set_xticks(np.arange(COLS))
    ax.set_yticks(np.arange(ROWS))

    for r in range(ROWS):
        for c in range(COLS):
            act = best_action[r, c]
            dy, dx = action_vec[act]
            ax.arrow(
                c,
                r,
                dx,
                dy,
                head_width=0.15,
                head_length=0.15,
                fc="white",
                ec="white",
                linewidth=1.2,
            )

    # draw holes/start/goal for reference
    for hr, hc in HOLES:
        ax.add_patch(Rectangle((hc - 0.5, hr - 0.5), 1, 1, facecolor="black"))
    ax.text(
        GOAL[1],
        GOAL[0],
        "G",
        ha="center",
        va="center",
        color="yellow",
        fontsize=14,
        weight="bold",
    )
    ax.text(
        START[1],
        START[0],
        "S",
        ha="center",
        va="center",
        color="cyan",
        fontsize=14,
        weight="bold",
    )

    ax.set_title(title)
    plt.show()


# ==================================================================
# 5️⃣  Console play‑through (greedy policy)
# ==================================================================
def play_greedy(
    agent: QLearningAgent,
    env: GridWorld,
    max_steps: int = 50,
    delay: float = 0.5,
) -> None:
    """
    Run ONE episode after training, always picking the greedy action,
    and print the board after each move.
    """
    # TODO:
    #   * reset the environment, render the initial board.
    #   * loop until done or max_steps:
    #         – find the best action for the current state (break ties randomly)
    #         – step the environment
    #         – render the board, sleep for `delay` seconds
    #   * after the loop print SUCCESS if the final position is GOAL,
    #     otherwise print FAILURE and how many steps were taken.
    pass


# ==================================================================
# 6️⃣  Matplotlib animation (greedy policy) – optional
# ==================================================================
def animate_greedy(
    agent: QLearningAgent,
    env: GridWorld,
    max_steps: int = 50,
    interval: int = 400,
) -> animation.FuncAnimation:
    """
    Build a Matplotlib FuncAnimation that shows the agent moving
    according to its greedy policy.
    """
    # TODO:
    #   * pre‑compute the greedy action for every state: greedy_action = np.argmax(agent.Q, axis=1)
    #   * create a heat‑map figure of max Q values (same code as plot_value_function)
    #   * draw holes/start/goal markers on the axes
    #   * add a red square (Rectangle) that will be moved each frame
    #   * define `init()` to place the marker at the start state
    #   * define `update(frame)`:
    #         – if episode finished, just return the marker
    #         – otherwise look up the greedy action for the current state,
    #           step the environment, move the marker, update counters
    #   * build and return the FuncAnimation object.
    pass


# ==================================================================
# 7️⃣  Main – train + visualise + play
# ==================================================================
if __name__ == "__main__":

    # --------------------------------------------------------------
    # 1️⃣  Train the agent
    # --------------------------------------------------------------
    # TODO: call train_agent() with the desired number of episodes etc.
    # Example (feel free to change the numbers):
    # trained = train_agent(num_episodes=500, max_steps=50, render_every=0)
    pass

    # --------------------------------------------------------------
    # 2️⃣  Show the final value‑function heat‑map
    # --------------------------------------------------------------
    # TODO: uncomment after you have a trained agent
    # plot_value_function(trained, title="Final learned value function")

    # --------------------------------------------------------------
    # 3️⃣  (Optional) draw the greedy policy arrows
    # --------------------------------------------------------------
    # plot_policy(trained, title="Greedy policy (arrows)")

    # --------------------------------------------------------------
    # 4️⃣  Console play‑through – watch the robot after training
    # --------------------------------------------------------------
    # env = GridWorld()
    # print("\n=== Console play‑through (greedy) ===\n")
    # play_greedy(trained, env, max_steps=30, delay=0.4)

    # --------------------------------------------------------------
    # 5️⃣  (Optional) Matplotlib animation
    # --------------------------------------------------------------
    # env_anim = GridWorld()
    # anim = animate_greedy(trained, env_anim, max_steps=30, interval=500)
    # plt.show()   # blocks until you close the window

    # --------------------------------------------------------------
    # End of script
    # --------------------------------------------------------------
