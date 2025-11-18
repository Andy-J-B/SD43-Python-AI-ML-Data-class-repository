# ----------------------------------------------------------------
#  MINIMAX AI – STUDENT SKELETON
# ----------------------------------------------------------------


# ----------------------------------------------------------------
# minimax -----------------------------------------------------------
# ----------------------------------------------------------------
def minimax(
    board: Board,  # the current game board (2‑D list)
    depth: int,  # recursion depth – how many moves ahead we are
    is_maximizing: bool,  # True → AI's turn, False → human's turn
    ai_player: str,  # symbol for the AI (e.g. "X")
    human_player: str,  # symbol for the human (e.g. "O")
) -> int:
    """
    Recursively evaluate the board and return a numeric score.
    *Higher* scores are better for the AI.
    You must:
    1️⃣ Detect if the game is over (win / loss / draw) and return the
       appropriate score (e.g. 10‑depth for an AI win, depth‑10 for a loss,
       0 for a draw).
    2️⃣ If the game is not finished, generate all legal moves.
    3️⃣ For each move:
        • Apply the move to the board.
        • Re‑curse with the opposite player (flip `is_maximizing`).
        • Undo the move (backtrack).
    4️⃣ Return the **maximum** score when `is_maximizing` is True,
       otherwise return the **minimum** score.
    """
    # ------------------------------------------------------------
    # 1️⃣  Check for a terminal state (win / loss / draw)
    # ------------------------------------------------------------
    # TODO: call `check_winner(board)` and compare with `ai_player` / `human_player`.
    # TODO: if a player has won, return 10‑depth or depth‑10 accordingly.
    # TODO: if the board is full with no winner, return 0.

    # ------------------------------------------------------------
    # 2️⃣  Branch out to the next level of the game tree
    # ------------------------------------------------------------
    if is_maximizing:
        # TODO: initialise `best` to -infinity.
        # TODO: loop over every empty cell (use `get_available_moves(board)`).
        # TODO: place `ai_player` in the cell, call `minimax` recursively
        #       with `depth+1` and `is_maximizing=False`, then undo the move.
        # TODO: keep the highest score seen and finally return it.
        pass
    else:
        # TODO: similar to the block above, but:
        #       * initialise `best` to +infinity,
        #       * place `human_player`,
        #       * recurse with `is_maximizing=True`,
        #       * keep the **lowest** score.
        pass


# ----------------------------------------------------------------
# ai_move -----------------------------------------------------------
# ----------------------------------------------------------------
def ai_move(board: Board, ai_player: str, human_player: str) -> Move:
    """
    Choose the best move for the AI by trying every legal move,
    scoring it with `minimax`, and returning the coordinates of the
    highest‑scoring move.
    You must:
    1️⃣ Initialise variables to keep track of the best score and best move.
    2️⃣ Iterate over all free squares (`get_available_moves(board)`).
    3️⃣ For each square:
        • Simulate the AI playing there.
        • Call `minimax(board, 0, False, ai_player, human_player)` to get its score.
        • Undo the simulated move.
        • If the score is better than the current best, store it and the move.
    4️⃣ Return the best move found.
    """
    # TODO: set `best_score` to -infinity and `best_move` to a dummy value.
    # TODO: loop over each (r, c) from `get_available_moves(board)`.
    # TODO: place `ai_player` on the board, call `minimax`, then revert the cell.
    # TODO: update `best_score` and `best_move` when a higher score appears.
    # TODO: after the loop, return `best_move`.
    pass
