# -------------------------------------------------------------
# gridworld_solution.py
# -------------------------------------------------------------
# 5×5 Frozen‑Lake Q‑learning implementation with optional visualisations.
# -------------------------------------------------------------
# Usage:
#   $ python gridworld_solution.py
#   (training runs, final heat‑map is shown, then a console play‑through)
#
#   To see the Matplotlib animation instead of the console version,
#   comment out `play_greedy(...)` and uncomment the `animate_greedy(...)`
#   block at the end of the file.
# -------------------------------------------------------------

import time
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle

# ------------------------------------------------------------------
# 0️⃣  Configuration
# ------------------------------------------------------------------
ROWS, COLS = 5, 5
START = (0, 0)
GOAL = (4, 4)
HOLES = {(1, 2), (2, 3), (3, 1), (3, 3)}  # you can change this later

ACTIONS = {
    0: (-1, 0),  # up
    1: (1, 0),  # down
    2: (0, -1),  # left
    3: (0, 1),  # right
}
N_ACTIONS = len(ACTIONS)


# ------------------------------------------------------------------
# 1️⃣  Environment
# ------------------------------------------------------------------
class GridWorld:
    """Simple frozen lake environment."""

    def __init__(self):
        self.reset()

    def reset(self) -> Tuple[int, int]:
        """Place the agent back at the start."""
        self.pos = START
        return self.pos

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """Move the agent, return (next_state, reward, done)."""
        dr, dc = ACTIONS[action]
        r, c = self.pos
        nr, nc = r + dr, c + dc

        # Keep the agent inside the grid
        if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS:
            nr, nc = r, c  # bounce off the wall

        self.pos = (nr, nc)

        # Rewards / termination
        if self.pos == GOAL:
            return self.pos, 1.0, True  # goal reached
        if self.pos in HOLES:
            return self.pos, -1.0, True  # fell in a hole
        return self.pos, -0.01, False  # ordinary step

    def get_valid_actions(self) -> List[int]:
        """All four actions are always allowed (step handles walls)."""
        return list(ACTIONS.keys())

    def render(self) -> None:
        """ASCII rendering – useful for quick debugging."""
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
# 2️⃣  Q‑Learning Agent
# ------------------------------------------------------------------
class QLearningAgent:
    """Tabular Q‑learning with ε‑greedy action selection."""

    def __init__(self, lr: float = 0.1, gamma: float = 0.99, eps: float = 0.2):
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.Q = np.zeros((ROWS * COLS, N_ACTIONS))

    # ------------------------------------------------------------------
    # Helper: state ↔ flat index
    # ------------------------------------------------------------------
    def _state_to_index(self, state: Tuple[int, int]) -> int:
        r, c = state
        return r * COLS + c

    # ------------------------------------------------------------------
    # Action selection (ε‑greedy)
    # ------------------------------------------------------------------
    def select_action(self, state: Tuple[int, int]) -> int:
        if np.random.rand() < self.eps:
            return np.random.choice(N_ACTIONS)

        idx = self._state_to_index(state)
        best_q = np.max(self.Q[idx])
        # break ties randomly
        candidates = np.where(self.Q[idx] == best_q)[0]
        return np.random.choice(candidates)

    # ------------------------------------------------------------------
    # Q‑table update (standard Q‑learning)
    # ------------------------------------------------------------------
    def update(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int],
        done: bool,
    ) -> None:
        s_idx = self._state_to_index(state)
        ns_idx = self._state_to_index(next_state)

        target = reward
        if not done:
            target += self.gamma * np.max(self.Q[ns_idx])

        self.Q[s_idx, action] += self.lr * (target - self.Q[s_idx, action])


# ------------------------------------------------------------------
# 3️⃣  Training loop
# ------------------------------------------------------------------
def train_agent(
    num_episodes: int = 300,
    max_steps: int = 50,
    render_every: int = 0,
) -> QLearningAgent:
    """
    Train the Q‑learning agent.

    Parameters
    ----------
    num_episodes : int
        How many episodes to run.
    max_steps : int
        Max number of steps per episode.
    render_every : int
        If >0, render the board & value‑map every `render_every` episodes.

    Returns
    -------
    QLearningAgent
        The trained agent.
    """
    env = GridWorld()
    agent = QLearningAgent(lr=0.1, gamma=0.99, eps=0.2)

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            # 1️⃣ Choose action
            action = agent.select_action(state)

            # 2️⃣ Apply to environment
            next_state, reward, done = env.step(action)

            # 3️⃣ Update Q‑table
            agent.update(state, action, reward, next_state, done)

            state = next_state
            steps += 1

        # optional visual feedback
        if render_every and ep % render_every == 0:
            print(f"\nEpisode {ep}")
            env.render()
            plot_value_function(agent, title=f"Episode {ep}")

    return agent


# ------------------------------------------------------------------
# 4️⃣  Plotting helpers
# ------------------------------------------------------------------
def plot_value_function(agent: QLearningAgent, title: str = "Value function"):
    """
    Visualise the learned state‑value as a 5×5 heat‑map.
    The colour corresponds to max_a Q(s,a).
    """
    V = np.max(agent.Q, axis=1).reshape((ROWS, COLS))

    plt.figure(figsize=(5, 4))
    plt.title(title)
    plt.imshow(V, cmap="viridis", origin="upper")
    plt.colorbar(label="max Q")
    plt.xticks(np.arange(COLS))
    plt.yticks(np.arange(ROWS))

    # annotate each cell (optional – comment out if noisy)
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
    """
    Draw arrows on top of the value heat‑map indicating the best action per state.
    """
    V = np.max(agent.Q, axis=1).reshape((ROWS, COLS))
    best_action = np.argmax(agent.Q, axis=1).reshape((ROWS, COLS))

    # Arrow vectors corresponding to action ids (dy, dx)
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

    # draw arrows
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

    # overlay holes/start/goal for context
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


# ------------------------------------------------------------------
# 5️⃣  Console play‑through (greedy policy)
# ------------------------------------------------------------------
def play_greedy(
    agent: QLearningAgent,
    env: GridWorld,
    max_steps: int = 50,
    delay: float = 0.5,
) -> None:
    """
    Run ONE episode after training using the greedy policy (no ε‑exploration)
    and print the board after each move.

    Parameters
    ----------
    agent : QLearningAgent
        The trained Q‑learning agent.
    env : GridWorld
        Fresh environment (reset will be called inside).
    max_steps : int
        Upper bound on steps – avoids infinite loops if something went wrong.
    delay : float
        Seconds to wait between frames (set to 0 for instant printing).
    """
    state = env.reset()
    env.render()
    time.sleep(delay)

    done = False
    steps = 0
    while not done and steps < max_steps:
        # choose best action (break ties randomly)
        s_idx = agent._state_to_index(state)
        best_q = np.max(agent.Q[s_idx])
        candidates = np.where(agent.Q[s_idx] == best_q)[0]
        action = np.random.choice(candidates)

        next_state, reward, done = env.step(action)

        env.render()
        time.sleep(delay)

        state = next_state
        steps += 1

    result = "SUCCESS!" if env.pos == GOAL else "FAILURE."
    print(f"Episode finished after {steps} steps – {result}")


# ------------------------------------------------------------------
# 6️⃣  Matplotlib animation (greedy policy)
# ------------------------------------------------------------------
def animate_greedy(
    agent: QLearningAgent,
    env: GridWorld,
    max_steps: int = 50,
    interval: int = 400,
) -> animation.FuncAnimation:
    """
    Return a Matplotlib animation that shows the agent moving through the
    grid according to its greedy policy.

    Parameters
    ----------
    agent : QLearningAgent
        Trained agent.
    env : GridWorld
        Fresh environment (will be reset inside).
    max_steps : int
        Maximum number of frames.
    interval : int
        Time between frames in milliseconds.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    # Pre‑compute greedy action for each state (single best action)
    greedy_action = np.argmax(agent.Q, axis=1)

    # ---- background heat‑map -------------------------------------------------
    V = np.max(agent.Q, axis=1).reshape((ROWS, COLS))
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(V, cmap="viridis", origin="upper")
    fig.colorbar(im, ax=ax, label="max Q")
    ax.set_xticks(np.arange(COLS))
    ax.set_yticks(np.arange(ROWS))
    ax.set_title("Greedy episode (animation)")

    # draw holes, start, goal
    for hr, hc in HOLES:
        ax.add_patch(
            Rectangle((hc - 0.5, hr - 0.5), 1, 1, facecolor="black", edgecolor="white")
        )
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

    # ---- agent marker (a red square) -----------------------------------------
    agent_marker = Rectangle((-0.5, -0.5), 1, 1, facecolor="red", edgecolor="white")
    ax.add_patch(agent_marker)

    # ---- animation state ------------------------------------------------------
    state = env.reset()
    steps = 0
    done = False

    def init():
        r, c = state
        agent_marker.set_xy((c - 0.5, r - 0.5))
        return (agent_marker,)

    def update(frame):
        nonlocal state, steps, done
        if steps >= max_steps or done:
            # freeze the last frame
            return (agent_marker,)

        # pick greedy action for current state
        s_idx = agent._state_to_index(state)
        action = greedy_action[s_idx]

        # step environment
        next_state, _, done = env.step(action)
        r, c = next_state
        agent_marker.set_xy((c - 0.5, r - 0.5))

        state = next_state
        steps += 1
        return (agent_marker,)

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=max_steps,
        interval=interval,
        blit=True,
        repeat=False,
    )
    return anim


# ------------------------------------------------------------------
# 7️⃣  Main – run training & visualisations
# ------------------------------------------------------------------
if __name__ == "__main__":
    # --------------------------------------------------------------
    # Train the agent
    # --------------------------------------------------------------
    trained = train_agent(
        num_episodes=500,  # feel free to increase
        max_steps=50,
        render_every=0,  # set >0 if you want intermediate heat‑maps
    )

    # --------------------------------------------------------------
    # Final value‑function heat‑map
    # --------------------------------------------------------------
    plot_value_function(trained, title="Final learned value function")

    # --------------------------------------------------------------
    # (Optional) Policy arrows overlay
    # --------------------------------------------------------------
    # plot_policy(trained, title="Greedy policy (arrows)")

    # --------------------------------------------------------------
    # Console play‑through – watch the robot after training
    # --------------------------------------------------------------
    env = GridWorld()
    print("\n=== Console play‑through (greedy) ===\n")
    play_greedy(trained, env, max_steps=30, delay=0.4)

    # --------------------------------------------------------------
    # (Optional) Matplotlib animation – uncomment the two lines below
    # --------------------------------------------------------------
    # env_anim = GridWorld()
    # anim = animate_greedy(trained, env_anim, max_steps=30, interval=500)
    # plt.show()  # blocks until you close the window

    # --------------------------------------------------------------
    # End of script
    # --------------------------------------------------------------
