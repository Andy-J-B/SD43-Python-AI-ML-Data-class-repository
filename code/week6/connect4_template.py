# connect4_template.py
# ------------------------------------------------------------
# CONNECT‑FOUR – AI Playground
# ------------------------------------------------------------
# Students will complete the functions marked with # TODO.
# DO NOT change function names or signatures – the test script
# imports these exact symbols.
# ------------------------------------------------------------

import numpy as np
import random
from copy import deepcopy

# ----------------------------------------------------------------------
# Game constants
# ----------------------------------------------------------------------
ROWS = 6
COLS = 7
WIN_LENGTH = 4  # 4 in a row wins

EMPTY = 0
PLAYER1 = 1  # will be controlled by minimax / α‑β
PLAYER2 = 2  # will be controlled by Monte‑Carlo or Q‑learning


# ----------------------------------------------------------------------
# Board representation + helper utilities
# ----------------------------------------------------------------------
class Board:
    """
    Simple Board class that stores the grid as a NumPy array.
    Rows are indexed from 0 (top) to ROWS‑1 (bottom); columns from 0‑COLS‑1.
    """

    def __init__(self):
        self.grid = np.zeros((ROWS, COLS), dtype=int)  # empty board
        self.last_move = None  # (row, col) of the most recent piece

    def copy(self):
        """Return a deep copy of the board (used by search algorithms)."""
        return deepcopy(self)

    # ------------------- 1️⃣ Legal moves -----------------------------
    def legal_moves(self):
        """Return a list of column indices (0 … COLS‑1) where a piece can be dropped."""
        # TODO: a column is legal if its topmost cell (row 0) is EMPTY.
        return [c for c in range(COLS) if self.grid[0, c] == EMPTY]


    # ------------------- 2️⃣ Apply a move ----------------------------
    def make_move(self, col, player):
        if self.grid[0,COLS] != 0:
            raise ValueError(f"{col} is full")
        for r in reversed(range(ROWS)):
            if self.grid[r,col] == 0:
                self.grid[r,col] = player
                self.last_move = (r,col)
                return self.last_move
            
        raise RuntimeError("make move failed")
        

    # ------------------- 3️⃣ Undo a move -----------------------------
    def undo_move(self, col):
        """
        Remove the topmost piece from column `col`.
        Used by minimax/α‑β search to backtrack.
        """
        # TODO: pop the highest non‑empty cell in column `col`.
        for r in range(ROWS):
            if self.grid[r, col] != EMPTY:
                self.grid[r, col] = EMPTY
                self.last_move = None
                return 
        raise ValueError
    # ------------------- 4️⃣ Terminal test --------------------------
    def is_full(self):
        """Return True if the board has no empty cells."""
        return not (self.grid == EMPTY).any()

    def check_winner(self):
        """
        Return PLAYER1, PLAYER2 if someone has a CONNECT‑FOUR,
        or EMPTY if no winner yet.
        """
        # TODO: scan rows, columns, and both diagonals for a run
        #       of WIN_LENGTH pieces belonging to the same player.
        def run_length(line):
            max_length = 0
            current_length = 0
            current_player = 0
            for p in line:
                if p == current_player and p != 0:
                    current_length = current_length + 1
                else:
                    current_player = p
                    if p != 0:
                      current_length = 1
                    else:
                        current_length = 0
                max_length = max(current_length, max_length)
            return max_length

    # ------------------- 5️⃣ Utility / evaluation ------------------
    def evaluate(self, player):
        """
        Heuristic evaluation of the board from the point of view of `player`.
        Positive = good for `player`, negative = good for opponent.
        Simple implementation: count open 2‑ and 3‑in‑a‑row patterns.
        """
        # TODO: implement a fast linear‑time eval (you can keep it extremely
        #       simple – just count the number of 2‑in‑a‑row with both ends open).
        raise NotImplementedError

    # ------------------- 6️⃣ Pretty printer -------------------------
    def __str__(self):
        """Return a human‑readable board (use . for empty, X for P1, O for P2)."""
        symbols = {EMPTY: ".", PLAYER1: "X", PLAYER2: "O"}
        rows = []
        for r in range(ROWS):
            row = " ".join(symbols[int(self.grid[r, c])] for c in range(COLS))
            rows.append(row)
        return "\n".join(rows) + "\n" + " ".join(map(str, range(COLS)))


# ----------------------------------------------------------------------
# 1️⃣ Minimax with Alpha‑Beta pruning (deterministic opponent)
# ----------------------------------------------------------------------
def alphabeta(board, depth, player, alpha=-np.inf, beta=np.inf):
    """
    Return (value, best_move) for the current `player` using α‑β pruning.
    Depth‑limited search – when depth == 0 or terminal state, return
    the board evaluation.

    Parameters
    ----------
    board : Board
    depth : int
    player : int (PLAYER1 or PLAYER2)
    alpha, beta : float (pruning bounds)

    Returns
    -------
    (value, move) : (float, int or None)
    """
    # TODO: implement standard α‑β minimax.
    #   * If board.is_full() or board.check_winner() != EMPTY --> terminal.
    #   * At depth == 0 use board.evaluate(player).
    #   * Iterate over board.legal_moves().
    #   * Remember the move that yields the best value.
    raise NotImplementedError


# ----------------------------------------------------------------------
# 2️⃣ Monte‑Carlo Tree Search (purely stochastic)
# ----------------------------------------------------------------------
class MCTSNode:
    """
    Very lightweight MCTS node – holds stats for a board state.
    For the classroom we use a *flat* Monte‑Carlo simulation: from the
    current board we run N random playouts for each legal move and pick
    the move with the highest win‑rate.
    """

    def __init__(self, board, player):
        """
        Initialise an MCTS node.
        * ``board`` – a ``Board`` instance representing the current game state.
        * ``player`` – the player who will move from this state (PLAYER1 or PLAYER2).

        The node stores a *copy* of the board to avoid mutating the original
        game state during simulations. It also prepares bookkeeping fields
        for the number of wins, total visits, and child nodes (unused in this
        flat implementation).
        """
        self.board = board.copy()
        self.player = player  # player to move at this node
        self.wins = 0
        self.visits = 0
        self.children = {}  # move -> MCTSNode

    # Simple rollout policy: both players choose random legal moves until terminal.
    @staticmethod
    def _random_playout(board, start_player):
        """
        Perform a completely random playout from the given board state.
        The method should:
            1. Repeatedly check if the game has ended (`board.check_winner()`
               or `board.is_full()`).
            2. If the game is not over, pick a legal move uniformly at random
               (`random.choice(board.legal_moves())`).
            3. Apply that move for the current player (`board.make_move(move, current)`).
            4. Switch the current player and keep looping.
        When a terminal state is reached, return the winner (PLAYER1,
        PLAYER2) or EMPTY for a draw.

        NOTE: Do **not** modify the original board object – the caller will
        pass in a copy that it can safely mutate.
        """
        raise NotImplementedError

    def simulate(self, simulations=50):
        """
        Run a flat Monte‑Carlo evaluation for each legal move.
        For each legal column:
            1. Clone the current board.
            2. Apply the candidate move for ``self.player``.
            3. Perform ``simulations`` random playouts starting with the
               opponent's turn.
            4. Count how many of those playouts end with ``self.player`` as
               the winner (store in ``wins``).

        After all moves have been evaluated, return the column with the
        highest win‑rate (wins / simulations). In case of a tie, break the
        tie randomly.
        """
        # TODO: implement the Monte‑Carlo evaluation as described.
        raise NotImplementedError


# ----------------------------------------------------------------------
# 3️⃣ Q‑Learning for Connect‑Four
#    – a tiny tabular agent that learns a policy by self‑play.
# ----------------------------------------------------------------------
class QLearningAgent:
    """
    Tabular Q‑learning agent.
    State is encoded as a string of 42 characters (0/1/2) – enough for an
    introductory demo (no deep net). For a class project you can keep the
    learning loop separate from the UI.
    """

    def __init__(self, player, alpha=0.5, gamma=0.9, epsilon=0.2):
        self.player = player
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration prob.
        self.Q = {}  # dict: state_str -> np.ndarray of Q-values per action

    @staticmethod
    def encode(board):
        """
        Convert the Board's NumPy grid into a compact, hashable string.
        Each cell contains 0 (EMPTY), 1 (PLAYER1) or 2 (PLAYER2).
        A simple way is to flatten the grid (row‑major order) and join the
        integer values:

            ''.join(str(int(v)) for v in board.grid.ravel())

        This string will be used as a dictionary key in ``self.Q``.
        """
        raise NotImplementedError

    def get_Q(self, state):
        """
        Return the Q‑value vector for a given ``state``.
        * ``state`` is the string produced by ``encode``.
        * If the state has never been seen, create a new entry:
          ``self.Q[state] = np.zeros(COLS)`` (one Q‑value for each column).
        The method must then return the NumPy array stored in the dictionary.
        """
        raise NotImplementedError

    def select_action(self, board):
        """
        Choose an action (column) using an ε‑greedy policy.
        * With probability ``self.epsilon`` choose a random legal move.
        * Otherwise, look up the Q‑values for the current state and pick the
          legal move with the highest Q‑value. If several moves share the
          maximal value, break ties randomly.
        * Return the selected column index (int).

        Hint: call ``self.encode(board)`` to obtain the state string,
        then ``self.get_Q(state)`` to retrieve the vector.
        """
        raise NotImplementedError

    def update(self, board, action, reward, next_board, done):
        """
        Apply the standard Q‑learning update:

            Q(s,a) ← Q(s,a) + α [ reward + γ * max_a' Q(s',a') – Q(s,a) ]

        * ``board`` is the state *before* taking ``action``.
        * ``next_board`` is the resulting state after the move.
        * ``reward`` is a scalar (e.g. +1 for a win, -1 for a loss, 0 otherwise).
        * ``done`` indicates whether ``next_board`` is terminal.

        Steps:
            1. Encode ``board`` and ``next_board``.
            2. Retrieve the Q‑vectors for both states via ``get_Q``.
            3. Compute the target value:
               - ``reward`` if ``done`` else
               - ``reward + self.gamma * max(Q_next)``.
            4. Update ``Q[state][action]`` in place using ``self.alpha``.
        No return value is required.
        """
        raise NotImplementedError

    def train_selfplay(self, episodes=5000, max_len=42):
        """
        Train the agent by letting two copies of the Q‑learning agent play
        against each other from the empty board.

        * ``episodes`` – number of games to play.
        * ``max_len`` – an upper bound on the number of moves per episode
          (the board can hold at most ROWS*COLS pieces, but a smaller
          bound can stop very long draws early).

        For each episode:
            1. Initialise a fresh ``Board``.
            2. Alternate between the two agents (one playing as PLAYER1,
               the other as PLAYER2) until the game ends or ``max_len`` moves
               have been made.
            3. After each move, compute the reward:
               - ``+1`` for the player who just won,
               - ``-1`` for the player who just lost,
               - ``0`` for all non‑terminal moves and draws.
            4. Call ``self.update`` (and the opponent's ``update``) with the
               observed transition.

        The method may optionally print progress every few thousand episodes,
        but no output is required for the grading script.
        """
        raise NotImplementedError


# ----------------------------------------------------------------------
# 4️⃣ Simple UI – text based
# ----------------------------------------------------------------------
def play_human_vs_ai():
    """
    Human (PLAYER2) vs AI (PLAYER1, using alphabeta).
    The function demonstrates the full loop:
        * print board
        * ask human for column
        * AI picks best move via alphabeta(depth=4)
    """
    board = Board()
    current = PLAYER1  # AI starts

    while True:
        print(board)
        winner = board.check_winner()
        if winner != EMPTY:
            print(f"WINNER: {'X' if winner == PLAYER1 else 'O'}")
            break
        if board.is_full():
            print("DRAW")
            break

        if current == PLAYER1:
            # --- AI move -------------------------------------------------
            value, move = alphabeta(board, depth=4, player=PLAYER1)
            print(f"AI (X) chooses column {move} (value={value:.2f})")
            board.make_move(move, PLAYER1)
        else:
            # --- Human move ----------------------------------------------
            valid = board.legal_moves()
            move = None
            while move not in valid:
                try:
                    move = int(input(f"Your turn (O). Choose column {valid}: "))
                except ValueError:
                    continue
            board.make_move(move, PLAYER2)

        # Switch player
        current = PLAYER1 if current == PLAYER2 else PLAYER2


def play_human_vs_mcts(simulations=60):
    """
    Human (PLAYER2) vs AI (PLAYER1) using Monte‑Carlo Tree Search.
    The game loop follows the same structure as ``play_human_vs_ai``,
    but replaces the minimax call with an MCTS call.

    Steps for the student:
        1. Print the board and check for terminal conditions.
        2. If it is the AI's turn:
            a. Create an ``MCTSNode`` with the current board and ``PLAYER1``.
            b. Call ``node.simulate(simulations)`` to obtain the AI's move.
            c. Apply the move with ``board.make_move(move, PLAYER1)``.
        3. If it is the human's turn:
            a. Prompt the user for a column (as in ``play_human_vs_ai``).
            b. Validate the input against ``board.legal_moves()``.
        4. Switch the current player and repeat until the game ends.

    The function should **not** modify the MCTSNode class – it only uses
    the public ``simulate`` method.
    """
    # TODO: implement the loop described above.
    raise NotImplementedError


def play_human_vs_qlearning(agent, epsilon=0.0):
    """
    Human (O) vs. a trained Q‑learning AI (X) on the command line.

    Parameters
    ----------
    agent   : QLearningAgent
              Must already have a populated ``.Q`` dictionary.
    epsilon : float, optional
              Exploration rate used only during this interactive game.
              0.0 → deterministic (always the best move);
              a small value (e.g. 0.05) lets the AI sometimes try a random
              legal move – useful for demo‑purposes.
    """
    # ------------------------------------------------------------------
    # 1️⃣  Initialise everything we need for the game
    # ------------------------------------------------------------------

    # 1a.  Use the supplied epsilon for this session (override the value
    #      the agent was trained with).
    # TODO: set the exploration rate of the agent
    # agent.epsilon = epsilon

    # 1b.  Create a brand‑new, empty board.
    # TODO: instantiate the Board class
    # board = ...

    # 1c.  Decide who moves first.  In this demo the AI (PLAYER1, “X”) starts.
    # TODO: set the variable that tracks whose turn it is
    # current = ...

    # 1d.  Greet the players.
    print("\n=== CONNECT‑FOUR – Human (O) vs. Q‑Learning AI (X) ===")

    # ------------------------------------------------------------------
    # 2️⃣  Main game loop – keep looping until someone wins or the board
    #     is full.
    # ------------------------------------------------------------------
    while True:
        # --------------------------------------------------------------
        # 2a.  Show the current board
        # --------------------------------------------------------------
        # TODO: print the board so the human can see the state
        # print(board)

        # --------------------------------------------------------------
        # 2b.  Check for a terminal condition (win or draw)
        # --------------------------------------------------------------
        # TODO: ask the board who the winner is (if any)
        # winner = ...

        # If somebody won → announce and break out of the loop
        # TODO: if winner != EMPTY: ... (print winner & break)

        # If the board is full → draw
        # TODO: if board.is_full(): ... (print draw & break)

        # --------------------------------------------------------------
        # 2c.  Decide which player makes the next move
        # --------------------------------------------------------------
        if current == PLAYER1:  # ----- AI (X) -----
            # TODO: ask the Q‑learning agent for its move on the current board
            # col = ...

            # Optional: show the AI's choice so the human can follow the game
            print(f"AI (X) chooses column {col}")

            # TODO: apply the move to the board (remember to pass PLAYER1)
            # board.make_move(col, PLAYER1)

        else:  # ----- Human (O) -----
            # TODO: retrieve the list of columns that are still legal
            # legal = ...

            # Prompt the user until they type a legal column number
            col = None
            while col not in legal:
                try:
                    # TODO: ask the user for input and convert to int
                    # col = int(input(f"Your turn (O). Choose column {legal}: "))
                    pass
                except ValueError:
                    # If they typed something that isn’t an int, just ask again
                    continue

            # TODO: record the human move on the board (use PLAYER2)
            # board.make_move(col, PLAYER2)

        # --------------------------------------------------------------
        # 2d.  Switch the player for the next iteration of the loop
        # --------------------------------------------------------------
        # TODO: toggle ``current`` between PLAYER1 and PLAYER2
        # current = ...

    # ------------------------------------------------------------------
    # 3️⃣  End of function – nothing to return (the game is printed)
    # ------------------------------------------------------------------


if __name__ == "__main__":
    # play_human_vs_ai()
    # play_human_vs_mcts()
    pass
