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
        if self.grid[0, COLS] != 0:
            raise ValueError(f"{col} is full")
        for r in reversed(range(ROWS)):
            if self.grid[r, col] == 0:
                self.grid[r, col] = player
                self.last_move = (r, col)
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

        for r in range(ROWS):
            if run_length(self.grid[r, :]) >= WIN_LENGTH:
                return self.grid[r, np.argmax(self.gird[r, :]) != EMPTY]

        for c in range(COLS):
            if run_length(self.grid[:, c]) >= WIN_LENGTH:
                return self.grid[np.argmax(self.gird[:, c], c) != EMPTY]

        # Diagonal (\)
        #
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

    # ------------------- 5️⃣ Utility / evaluation ------------------
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
        current = start_player
        while True:
            checkwin = board.check_winner()
            if checkwin != EMPTY:
                return checkwin
            if board.is_full():
                return EMPTY  # draw
            else:
                move = random.choice(board.legal_moves())
                board.make_move(move, current)
                current = PLAYER1 if current == PLAYER2 else PLAYER2
        """
        Perform a completely random playout from the given board state.
        The method 
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
        legal_moves = self.board.legal_moves()
        win_counts = {}
        for move in legal_moves:
            win_counts[move] = 0
        for move in legal_moves:
            for _ in range(simulations):
                trial = board.copy()
                trial.make_move(move, self.player)
                winner = self._random_playout(
                    trial, PLAYER1 if self.player == PLAYER2 else PLAYER2
                )
                if winner = self.player:
                    win_counts[move] += 1
        best_move = max(legal_moves, key = lambda move : win_counts[move]   )


# ----------------------------------------------------------------------
# 3️⃣ Q‑Learning for Connect‑Four
#    – a tiny tabular agent that learns a policy by self‑play.
# ----------------------------------------------------------------------
class QLearningAgent:
    """
    Tabular Q‑learning agent for Connect‑Four.

    *State representation* – the board (a 6 × 7 NumPy array) is flattened
    row‑major and each cell is turned into a character ``'0'`` (empty),
    ``'1'`` (PLAYER1) or ``'2'`` (PLAYER2).  The resulting 42‑character
    string is used as a dictionary key.

    *Action space* – the 7 columns (indices 0 … 6).  An entry in the
    Q‑table is therefore a length‑7 NumPy vector.

    The class implements the full learning loop (self‑play) plus the
    ordinary Q‑learning utilities needed by the UI.
    """

    # ------------------------------------------------------------------
    # 1️⃣  Construction
    # ------------------------------------------------------------------
    def __init__(self, player, alpha=0.5, gamma=0.9, epsilon=0.2):
        """
        Parameters
        ----------
        player  : int   – which side this agent plays (PLAYER1 or PLAYER2)
        alpha   : float – learning‑rate (0 < α ≤ 1)
        gamma   : float – discount factor (0 ≤ γ ≤ 1)
        epsilon : float – probability of taking a random (exploratory) move
        """
        self.player = player
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration probability
        # Q‑table:  state string → np.ndarray of shape (COLS,)
        self.Q = {}

    # ------------------------------------------------------------------
    # 2️⃣  Encode a board into a hashable string
    # ------------------------------------------------------------------
    @staticmethod
    def encode(board):
        """
        Convert the Board's NumPy grid into a compact, hashable string.
        Each cell contains 0 (EMPTY), 1 (PLAYER1) or 2 (PLAYER2).

        Example (tiny 2×3 board):
            [[0, 1, 0],
             [2, 0, 1]]  →  "010201"

        The result will be used as a key in ``self.Q``.
        """
        # TODO: flatten the array (row‑major) and join the integer values.
        #       Hint:  board.grid.ravel()  gives a 1‑D view.
        boardstr = board.grid.ravel()
        stringparts = []
        for cell in boardstr:
            intvalue =  int(cell)
            strvalue = str(intvalue)
            stringparts.append(strvalue)

        return "".join(stringparts)


    # ------------------------------------------------------------------
    # 3️⃣  Get (or create) the Q‑vector for a state
    # ------------------------------------------------------------------
    def get_Q(self, state):
        
        """
        Return the Q‑value vector for a given ``state``.
        If the state has never been seen, initialise an entry of zeros.

        Parameters
        ----------
        state : str   – the 42‑character string returned by ``encode``.

        Returns
        -------
        np.ndarray of shape (COLS,) – mutable array of Q‑values.
        """
        # TODO: if ``state`` not yet a key → create ``np.zeros(COLS)``.
        # TODO: return the stored array.
        if state not in self.Q:
            self.Q[state] = np.zeros(COLS)
        return self.Q[state]
    # ------------------------------------------------------------------
    # 4️⃣  ε‑greedy action selection
    # ------------------------------------------------------------------
    def select_action(self, board):
        """
        Choose a column for the current board.

        * With probability ``self.epsilon`` pick a random legal column
          (exploration).
        * Otherwise pick the legal column with the highest Q‑value
          (exploitation).  If several legal columns share the maximal
          value, break ties randomly.

        Returns
        -------
        int – the chosen column index (0 … COLS‑1).
        """
        # TODO: 1️⃣ obtain the list of legal columns:  board.legal_moves()
        # TODO: 2️⃣ encode the board & fetch its Q‑vector
        # TODO: 3️⃣ with probability ε return a random legal move
        # TODO: 4️⃣ otherwise select the legal move(s) with maximal Q,
        #          break ties randomly, and return the chosen column.
        legal_moves = board.legal_moves()
        state = self.encode(board)
        q = self.get_Q(state)
        if random.Random < self.epsilon:
            return random.choice(legal_moves)
        best_q = max(q[c] for c in legal_moves)
        best_moves = [c for c in legal_moves if q[c] == best_q]
        return random.choice(best_moves)


    # ------------------------------------------------------------------
    # 5️⃣  Q‑learning update rule
    # ------------------------------------------------------------------
    def update(self, board, action, reward, next_board, done):
        """
        Apply the standard Q‑learning update:

            Q(s,a) ← Q(s,a) + α [ reward + γ·max_a' Q(s',a') – Q(s,a) ]

        Parameters
        ----------
        board      : Board   – state *before* taking ``action``.
        action     : int or None – column that was played.
                       If ``None`` the call is a *virtual* update for the
                       opponent (no Q‑value is modified).
        reward     : float   – immediate scalar reward (+1 win, -1 loss, 0 otherwise).
        next_board : Board   – state *after* the move.
        done       : bool    – True if ``next_board`` is terminal.
        """
        # If this is a virtual update (called for the opponent) we do nothing.
        # TODO: if action is None → return immediately.

        # TODO: 1️⃣ encode the current state (s) and the next state (s')
        # TODO: 2️⃣ retrieve the Q‑vectors for s and s' via ``get_Q``.
        #          If ``done`` use a zero‑vector for the next state.
        # TODO: 3️⃣ compute the target:
        #          target = reward                     (if done)
        #          target = reward + γ * max(Q_next)   (otherwise)
        # TODO: 4️⃣ perform the incremental update:
        #          Q[s][action] += α * (target - Q[s][action])
        if (action == None):
            return
        state = self.encode(board)
        state_next = self.encode(next_board)
        q = self.get_Q(state)
        next_q = self.get_Q(state_next) if not done else np.zeros(COLS)
        target = reward + self.gamma * np.max(next_q)
        q[action] += self.alpha * (target - q[action])

    def train_selfplay(self, episodes=5000, max_len=42):
        """
        Train the agent by letting two copies of the Q‑learning agent play
        against each other from the empty board.

        Parameters
        ----------
        episodes : int – number of games to generate.
        max_len : int – upper bound on the number of moves per episode
                       (prevents extremely long draws).

        Training loop (what you have to implement)
        ------------------------------------------------------------
        for each episode:
            1️⃣  Create a fresh ``Board`` and set ``current = PLAYER1``.
            2️⃣  Store a copy of the board *before* the first move
                (this will be the ``prev_board`` used for updates).
            3️⃣  While the game is not finished:
                a)  Choose which agent is to move this turn
                    (the one whose ``player`` matches ``current``).
                b)  Let that agent pick an action with ``select_action``.
                c)  Apply the action to the board (``board.make_move``).
                d)  Detect terminal condition:
                    - ``winner = board.check_winner()``
                    - ``terminal = winner != EMPTY or board.is_full()``
                e)  Compute zero‑sum rewards:
                       if terminal:
                           reward_act =  +1 for the player who just won,
                                           -1 for the opponent.
                       else:
                           reward_act = reward_opp = 0
                f)  Create ``next_state`` = ``board.copy()``.
                g)  **Real update** for the acting agent:
                       ``act_agent.update(prev_board, move,
                                          reward_act, next_state,
                                          done=terminal)``.
                h)  **Virtual update** for the opponent:
                       ``opp_agent.update(prev_board, None,
                                          reward_opp, next_state,
                                          done=terminal)``.
                i)  Set ``prev_board = next_state`` and switch ``current``.
            4️⃣  (optional) print progress every few thousand episodes.

        The method does **not** return anything; after it finishes the
        attribute ``self.Q`` contains the learned Q‑table.
        """
        # ----------------------------------------------------------------
        # 0️⃣  Create the opponent (same hyper‑parameters, opposite colour)
        # ----------------------------------------------------------------
        # TODO: instantiate ``opponent = QLearningAgent(opp_player, …)``

        # ----------------------------------------------------------------
        # 1️⃣  Main episode loop
        # ----------------------------------------------------------------
        # TODO: for ep in range(episodes): …
        #       – decay epsilon here if you want an exploration schedule
        #       – reset the board, set current player, store prev_board
        #       – run the inner while‑not‑done loop as described above
        opponentplayer = PLAYER1 if self.player == PLAYER2 else PLAYER2
        opponent = QLearningAgent(opponentplayer, alpha=self.alpha, gamma=self.gamma, epsilon=0.5)
        max_length = 42
        epsilon_end=0.05
        decay_rate = 0.9995
        for ep in range(episodes):
            current = PLAYER1
            board = Board()
            done = False
            previous_board = board.copy()
            epsilon = max(epsilon_end, epsilon * decay_rate)
            self.epsilon = opponent.epsilon = epsilon
            while not done:
                agent = self if current == self.player else opponent
                opponent_agent = opponent if agent == self else self
                move = agent.select_action(board)
                board.make_move(move, current)
                winner = board.check_winner()
                terminal = winner != EMPTY or board.is_full()
                if terminal:
                    reward_agent = 1 if winner == agent.player else -1
                    reward_opponent = -reward_agent
                else:
                    reward_agent = 0 
                    reward_opponent = 0
                next_state = board.copy()
                agent.update(previous_board, move, reward_agent, next_state, done=terminal)
                opponent_agent.update(previous_board, None, reward_opponent, next_state, done=terminal)
                previous_board = next_state 
                current = PLAYER1 if current == PLAYER2 else PLAYER2
                done = terminal
            if (ep + 1) % 1000 == 0:
                print(f"Episode {ep + 1} completed.")
        return 
    
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
    board = Board() 
    current = PLAYER1 #oh lol
    
    while True:
        print(board)
        winner = board.check_winner() 
        if winner != EMPTY:
            print(f"Winner: {'X (MCTS)' if winner == PLAYER1 else 'O (Human)'}")
            break
        else:
            if board.is_full():
                print("Draw.")
                break
        
        if current == PLAYER1:
            mcts = MCTSNode(board, player) 
            move = mcts.simulate(simulations=simulations)
            # ty

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
    agent.epsilon = epsilon
    # 1b.  Create a brand‑new, empty board.
    # TODO: instantiate the Board class
    # board = ...
    board = Board()
    # 1c.  Decide who moves first.  In this demo the AI (PLAYER1, “X”) starts.
    # TODO: set the variable that tracks whose turn it is
    # current = ...
    current = PLAYER1
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
        print(board)
        # --------------------------------------------------------------
        # 2b.  Check for a terminal condition (win or draw)
        # --------------------------------------------------------------
        # TODO: ask the board who the winner is (if any)
        # winner = ...
        winner = board.check_winner()
        if winner != EMPTY:
            break
        if board.is_full():
            break
        
        # If somebody won → announce and break out of the loop
        # TODO: if winner != EMPTY: ... (print winner & break)
        if current == PLAYER1:
            col = agent.select_action(board)
            board.make_move(col, PLAYER1)
        else:
            legal = board.legal_moves()
            col = None
            while col not in legal:
                try:
                    col = int(input(f"Your turn. Choose: {legal}"))
                except ValueError:
                    continue
            board.make_move(col, PLAYER2)
        current = PLAYER1 if current == PLAYER2 else PLAYER2
        



if __name__ == "__main__":
    # play_human_vs_ai()
    # play_human_vs_mcts()
    play_human_vs_qlearning()
    pass
