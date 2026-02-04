# connect4_solution.py
# ------------------------------------------------------------
# CONNECT‑FOUR – Full solution (reference)
# ------------------------------------------------------------

import numpy as np
import random
from copy import deepcopy

# ----------------------------------------------------------------------
# Game constants
# ----------------------------------------------------------------------
ROWS = 6
COLS = 7
WIN_LENGTH = 4

EMPTY = 0
PLAYER1 = 1  # Minimax / α‑β
PLAYER2 = 2  # Monte‑Carlo or Q‑learning / Human


# ----------------------------------------------------------------------
# Board class (unchanged from template, but fully implemented)
# ----------------------------------------------------------------------
class Board:
    def __init__(self):
        self.grid = np.zeros((ROWS, COLS), dtype=int)
        self.last_move = None

    def copy(self):
        return deepcopy(self)

    # --------------------------------------------------------------
    def legal_moves(self):
        return [c for c in range(COLS) if self.grid[0, c] == EMPTY]

    # --------------------------------------------------------------
    def make_move(self, col, player):
        if self.grid[0, col] != EMPTY:
            raise ValueError(f"Column {col} is full")
        # Find the lowest empty row
        for row in reversed(range(ROWS)):
            if self.grid[row, col] == EMPTY:
                self.grid[row, col] = player
                self.last_move = (row, col)
                return (row, col)
        # Should never reach here
        raise RuntimeError("make_move failed unexpectedly")

    # --------------------------------------------------------------
    def undo_move(self, col):
        # Remove the topmost piece in column `col`
        for row in range(ROWS):
            if self.grid[row, col] != EMPTY:
                self.grid[row, col] = EMPTY
                self.last_move = None
                return
        raise ValueError(f"Cannot undo – column {col} is already empty")

    # --------------------------------------------------------------
    def is_full(self):
        return not (self.grid == EMPTY).any()

    # --------------------------------------------------------------
    def check_winner(self):
        # Helper to check runs
        def run_length(line):
            max_len = 0
            cur_len = 0
            cur_player = EMPTY
            for p in line:
                if p == cur_player and p != EMPTY:
                    cur_len += 1
                else:
                    cur_player = p
                    cur_len = 1 if p != EMPTY else 0
                max_len = max(max_len, cur_len)
            return max_len

        # Horizontal
        for r in range(ROWS):
            if run_length(self.grid[r, :]) >= WIN_LENGTH:
                return self.grid[r, np.argmax(self.grid[r, :] != EMPTY)]

        # Vertical
        for c in range(COLS):
            if run_length(self.grid[:, c]) >= WIN_LENGTH:
                return self.grid[np.argmax(self.grid[:, c] != EMPTY), c]

        # Diagonal (\)
        for offset in range(-ROWS + 1, COLS):
            diag = np.diagonal(self.grid, offset=offset)
            if run_length(diag) >= WIN_LENGTH:
                idx = np.argmax(diag != EMPTY)
                r = max(0, -offset) + idx
                c = max(0, offset) + idx
                return self.grid[r, c]

        # Diagonal (/)
        flipped = np.fliplr(self.grid)
        for offset in range(-ROWS + 1, COLS):
            diag = np.diagonal(flipped, offset=offset)
            if run_length(diag) >= WIN_LENGTH:
                idx = np.argmax(diag != EMPTY)
                r = max(0, -offset) + idx
                c = COLS - 1 - (max(0, offset) + idx)
                return self.grid[r, c]

        return EMPTY

    # --------------------------------------------------------------
    def evaluate(self, player):
        """
        Very simple linear heuristic:
        +10 for each 3‑in‑a‑row (open at both ends),
        +1  for each 2‑in‑a‑row (open at both ends).
        Subtract the same values for the opponent.
        """
        opponent = PLAYER1 if player == PLAYER2 else PLAYER2

        def count_patterns(p):
            count2 = 0
            count3 = 0

            # Helper to examine a line (list of cells)
            def scan(line):
                nonlocal count2, count3
                n = len(line)
                for i in range(n - WIN_LENGTH + 1):
                    window = line[i : i + WIN_LENGTH]
                    # Count pieces of player p
                    if window.count(p) == 3 and window.count(EMPTY) == 1:
                        # ensure empties are on the ends (open)
                        if (i > 0 and line[i - 1] == EMPTY) or (
                            i + WIN_LENGTH < n and line[i + WIN_LENGTH] == EMPTY
                        ):
                            count3 += 1
                    if window.count(p) == 2 and window.count(EMPTY) == 2:
                        # open on both ends
                        left_ok = i > 0 and line[i - 1] == EMPTY
                        right_ok = i + WIN_LENGTH < n and line[i + WIN_LENGTH] == EMPTY
                        if left_ok and right_ok:
                            count2 += 1

            # Horizontal
            for r in range(ROWS):
                scan(self.grid[r, :])
            # Vertical
            for c in range(COLS):
                scan(self.grid[:, c])
            # Diagonal (\)
            for offset in range(-ROWS + 1, COLS):
                scan(np.diagonal(self.grid, offset=offset))
            # Diagonal (/)
            flipped = np.fliplr(self.grid)
            for offset in range(-ROWS + 1, COLS):
                scan(np.diagonal(flipped, offset=offset))

            return count2, count3

        my2, my3 = count_patterns(player)
        opp2, opp3 = count_patterns(opponent)

        return (my3 - opp3) * 10 + (my2 - opp2)

    # --------------------------------------------------------------
    def __str__(self):
        sym = {EMPTY: ".", PLAYER1: "X", PLAYER2: "O"}
        rows = [
            " ".join(sym[int(self.grid[r, c])] for c in range(COLS))
            for r in range(ROWS)
        ]
        return "\n".join(rows) + "\n" + " ".join(map(str, range(COLS)))


# ----------------------------------------------------------------------
# 1️⃣ α‑β minimax (deterministic AI)
# ----------------------------------------------------------------------
def alphabeta(board, depth, player, alpha=-np.inf, beta=np.inf):
    winner = board.check_winner()
    if winner == player:
        return (np.inf, None)  # win for player
    if winner != EMPTY:
        return (-np.inf, None)  # loss for player
    if board.is_full():
        return (0, None)  # draw

    if depth == 0:
        return (board.evaluate(player), None)

    best_move = None
    if player == PLAYER1:
        # Maximising player
        value = -np.inf
        for move in board.legal_moves():
            board.make_move(move, player)
            child_val, _ = alphabeta(board, depth - 1, PLAYER2, alpha, beta)
            board.undo_move(move)
            if child_val > value:
                value, best_move = child_val, move
            alpha = max(alpha, value)
            if beta <= alpha:
                break  # β‑cutoff
        return (value, best_move)
    else:
        # Minimising player (opponent)
        value = np.inf
        for move in board.legal_moves():
            board.make_move(move, player)
            child_val, _ = alphabeta(board, depth - 1, PLAYER1, alpha, beta)
            board.undo_move(move)
            if child_val < value:
                value, best_move = child_val, move
            beta = min(beta, value)
            if beta <= alpha:
                break  # α‑cutoff
        return (value, best_move)


# ----------------------------------------------------------------------
# 2️⃣ Monte‑Carlo Tree Search (simple flat version)
# ----------------------------------------------------------------------
class MCTSNode:
    def __init__(self, board, player):
        self.board = board.copy()
        self.player = player

    @staticmethod
    def _random_playout(board, start_player):
        """Return winner (PLAYER1 / PLAYER2) or EMPTY for draw."""
        cur = start_player
        while True:
            win = board.check_winner()
            if win != EMPTY:
                return win
            if board.is_full():
                return EMPTY
            move = random.choice(board.legal_moves())
            board.make_move(move, cur)
            cur = PLAYER1 if cur == PLAYER2 else PLAYER2

    def simulate(self, simulations=80):
        """
        Run Monte‑Carlo simulations for every legal move and return the move
        with the highest win‑rate for `self.player`.
        """
        legal = self.board.legal_moves()
        win_counts = {m: 0 for m in legal}
        for move in legal:
            for _ in range(simulations):
                # Clone board, make the candidate move, then roll out
                trial = self.board.copy()
                trial.make_move(move, self.player)
                winner = self._random_playout(
                    trial, PLAYER1 if self.player == PLAYER2 else PLAYER2
                )
                if winner == self.player:
                    win_counts[move] += 1
        # Choose move with max win‑rate (break ties randomly)
        best_move = max(legal, key=lambda m: win_counts[m])
        return best_move


# ----------------------------------------------------------------------
# 3️⃣ Tabular Q‑Learning (optional, can be omitted from live‑coding)
# ----------------------------------------------------------------------
class QLearningAgent:
    def __init__(self, player, alpha=0.5, gamma=0.9, epsilon=0.2):
        self.player = player
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = {}  # state_str → np.ndarray[COLS]

    @staticmethod
    def encode(board):
        """Flatten the 6×7 grid to a string of 0/1/2."""
        # 1. Flatten the 2D grid into a 1D sequence
        flattened_values = board.grid.ravel()

        # 2. Initialize an empty list to store the string pieces
        # (Using a list is faster than adding strings together one by one)
        string_parts = []

        for v in flattened_values:
            # 3. Convert the value to an int, then to a string
            int_value = int(v)
            str_value = str(int_value)

            # 4. Add it to our collection
            string_parts.append(str_value)

        # 5. Combine everything into a single string
        return "".join(string_parts)

    def get_Q(self, state):
        """Return (and create if needed) the Q‑vector for a state."""
        if state not in self.Q:
            self.Q[state] = np.zeros(COLS)
        return self.Q[state]

    def select_action(self, board):
        """ε‑greedy policy – returns a legal column."""
        legal = board.legal_moves()
        state = self.encode(board)
        q = self.get_Q(state)

        if random.random() < self.epsilon:
            return random.choice(legal)  # explore
        # exploitation – pick the highest Q among *legal* moves
        best_q = max(q[c] for c in legal)
        best_moves = [c for c in legal if q[c] == best_q]
        return random.choice(best_moves)  # break ties randomly

    def update(self, board, action, reward, next_board, done):
        """Standard Q‑learning update.  `action` may be None (virtual update)."""
        if (
            action is None
        ):  # nothing to adjust – only used for the opponent’s virtual step
            return

        s = self.encode(board)
        s_next = self.encode(next_board)

        q = self.get_Q(s)
        q_next = self.get_Q(s_next) if not done else np.zeros(COLS)

        target = reward + self.gamma * np.max(q_next)
        q[action] += self.alpha * (target - q[action])

    # ------------------------------------------------------------------
    # Self‑play with dual‑updates + ε‑decay
    # ------------------------------------------------------------------
    def train_selfplay(
        self,
        episodes=50000,
        max_len=42,
        epsilon_start=0.5,
        epsilon_end=0.05,
        decay_rate=0.9995,
    ):
        """Two agents (self + opponent) play against each other."""
        opp_player = PLAYER1 if self.player == PLAYER2 else PLAYER2
        opponent = QLearningAgent(
            opp_player,
            alpha=self.alpha,
            gamma=self.gamma,
            epsilon=epsilon_start,
        )

        epsilon = epsilon_start

        for ep in range(episodes):
            board = Board()
            current = PLAYER1  # always let PLAYER1 start
            done = False

            # start of the episode – store the state *before* the first move
            prev_board = board.copy()

            # decay ε
            epsilon = max(epsilon_end, epsilon * decay_rate)
            self.epsilon = opponent.epsilon = epsilon

            while not done:
                # ------------------------------------------------------
                # Choose which agent moves this turn
                # ------------------------------------------------------
                act_agent = self if current == self.player else opponent
                opp_agent = opponent if act_agent is self else self

                move = act_agent.select_action(board)
                board.make_move(move, current)

                # ------------------------------------------------------
                # Examine the result
                # ------------------------------------------------------
                winner = board.check_winner()
                terminal = winner != EMPTY or board.is_full()

                # ------------------------------------------------------
                # Rewards (zero‑sum)
                # ------------------------------------------------------
                if terminal:
                    r_act = 1 if winner == act_agent.player else -1
                    r_opp = -r_act
                else:
                    r_act = r_opp = 0

                # ------------------------------------------------------
                # Q‑updates (both agents)
                # ------------------------------------------------------
                next_state = board.copy()

                # real update for the player that actually acted
                act_agent.update(prev_board, move, r_act, next_state, done=terminal)

                # virtual update for the opponent – no action of its own
                opp_agent.update(prev_board, None, r_opp, next_state, done=terminal)

                # ------------------------------------------------------
                # Prepare next iteration
                # ------------------------------------------------------
                prev_board = next_state
                current = PLAYER1 if current == PLAYER2 else PLAYER2
                done = terminal

            # optional progress report
            if (ep + 1) % 5000 == 0:
                print(f"Episode {ep + 1:,}/{episodes:,} – ε={epsilon:.3f}")

        print(f"Training finished after {episodes:,} episodes.")


# ----------------------------------------------------------------------
# 4️⃣ Simple interactive demo (human vs. AI)
# ----------------------------------------------------------------------
def play_human_vs_ai(depth=4):
    print("\n=== CONNECT FOUR – Human (O) vs. AI (X) ===")
    board = Board()
    current = PLAYER1  # AI starts

    while True:
        print(board)
        winner = board.check_winner()
        if winner != EMPTY:
            print(f"Winner: {'X (AI)' if winner == PLAYER1 else 'O (Human)'}")
            break
        if board.is_full():
            print("Game ends in a draw.")
            break

        if current == PLAYER1:
            # AI (minimax)
            val, move = alphabeta(board, depth=depth, player=PLAYER1)
            print(f"AI (X) picks column {move} (score={val:.1f})")
            board.make_move(move, PLAYER1)
        else:
            # Human turn
            legal = board.legal_moves()
            move = None
            while move not in legal:
                try:
                    move = int(input(f"Your turn (O). Choose column {legal}: "))
                except ValueError:
                    continue
            board.make_move(move, PLAYER2)

        # Switch player
        current = PLAYER1 if current == PLAYER2 else PLAYER2


def play_human_vs_mcts(simulations=60):
    print("\n=== CONNECT FOUR – Human (O) vs. MCTS (X) ===")
    board = Board()
    current = PLAYER1

    while True:
        print(board)
        winner = board.check_winner()
        if winner != EMPTY:
            print(f"Winner: {'X (MCTS)' if winner == PLAYER1 else 'O (Human)'}")
            break
        if board.is_full():
            print("Draw.")
            break

        if current == PLAYER1:
            mcts = MCTSNode(board, PLAYER1)
            move = mcts.simulate(simulations=simulations)
            print(f"MCTS (X) picks column {move} (after {simulations} sims)")
            board.make_move(move, PLAYER1)
        else:
            legal = board.legal_moves()
            move = None
            while move not in legal:
                try:
                    move = int(input(f"Your turn (O). Choose column {legal}: "))
                except ValueError:
                    continue
            board.make_move(move, PLAYER2)

        current = PLAYER1 if current == PLAYER2 else PLAYER2


def play_human_vs_qlearning(agent, epsilon=0.0):
    """
    Interactive command‑line game where a human (O) plays against a
    trained Q‑learning agent (X).

    Parameters
    ----------
    agent   : QLearningAgent
              Must already have been trained (its `.Q` dict filled).
    epsilon : float, optional
              Exploration rate for the AI during the game.
              Keep it 0.0 for a deterministic opponent; a small value
              (e.g. 0.05) makes the AI occasionally try a different move.
    """
    # ------------------------------------------------------------------
    # 1.  Prepare the agent and the board
    # ------------------------------------------------------------------
    agent.epsilon = epsilon  # override the training epsilon
    board = Board()
    current = PLAYER1  # X (the AI) starts

    print("\n=== CONNECT‑FOUR – Human (O) vs. Q‑Learning AI (X) ===")
    while True:
        # ------------------------------------------------------------------
        # 2.  Show board & test for terminal state
        # ------------------------------------------------------------------
        print(board)
        winner = board.check_winner()
        if winner != EMPTY:
            print(f"Winner: {'X (AI)' if winner == PLAYER1 else 'O (Human)'}")
            break
        if board.is_full():
            print("Game ends in a draw.")
            break

        # ------------------------------------------------------------------
        # 3.  Choose and apply a move
        # ------------------------------------------------------------------
        if current == PLAYER1:  # AI turn
            col = agent.select_action(board)  # <-- uses the trained Q‑table
            print(f"AI (X) chooses column {col}")
            board.make_move(col, PLAYER1)
        else:  # Human turn
            legal = board.legal_moves()
            col = None
            while col not in legal:
                try:
                    col = int(input(f"Your turn (O). Choose column {legal}: "))
                except ValueError:
                    continue
            board.make_move(col, PLAYER2)

        # ------------------------------------------------------------------
        # 4.  Switch player and loop
        # ------------------------------------------------------------------
        current = PLAYER1 if current == PLAYER2 else PLAYER2


if __name__ == "__main__":
    # Choose one demo to run (uncomment):
    # play_human_vs_ai(depth=4)
    # play_human_vs_mcts(simulations=500)

    # Example of training a Q‑learning agent (optional):
    agent = QLearningAgent(player=PLAYER1, epsilon=0.3)
    agent.train_selfplay(episodes=50)
    play_human_vs_qlearning(agent, epsilon=0.0)

    pass
