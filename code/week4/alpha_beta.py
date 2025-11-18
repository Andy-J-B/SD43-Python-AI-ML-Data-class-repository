# ----------------------------------------------------------------
#  MINIMAX WITH ALPHA‑BETA PRUNING – STUDENT SKELETON
# ----------------------------------------------------------------


def minimax_alpha_beta(
    board: Board,  # current board state
    depth: int,  # recursion depth
    alpha: float,  # best score maximiser can guarantee so far
    beta: float,  # best score minimiser can guarantee so far
    is_maximizing: bool,  # True → AI's turn, False → human's turn
    ai_player: str,  # AI symbol (e.g. "X")
    human_player: str,  # Human symbol (e.g. "O")
) -> int:
    """
    MINIMAX WITH ALPHA‑BETA PRUNING

    Return an integer score (+10 for AI win, –10 for Human win, 0 for draw).
    Implement the algorithm exactly as described in the docstring:
    * check terminal states,
    * recurse while updating α and β,
    * cut off branches when α ≥ β (for maximiser) or β ≤ α (for minimiser).
    """
    # ------------------------------------------------------------
    # 1️⃣  Terminal state handling
    # ------------------------------------------------------------
    # TODO: call `check_winner(board)`. If the winner is `ai_player`
    #       return 10 - depth, if the winner is `human_player` return depth - 10.
    # TODO: if `is_draw(board)` is True, return 0.

    # ------------------------------------------------------------
    # 2️⃣  Recursive exploration with alpha‑beta pruning
    # ------------------------------------------------------------
    if is_maximizing:
        # TODO: initialise `value` to -infinity.
        # TODO: loop over each empty square from `get_available_moves(board)`.
        #       * Place `ai_player` on the board.
        #       * Call `minimax_alpha_beta` recursively with depth+1,
        #         the current α, β, and `is_maximizing=False`.
        #       * Undo the move.
        #       * Update `value = max(value, score)`.
        #       * Update `alpha = max(alpha, value)`.
        #       * If `alpha >= beta` break the loop (beta cut‑off).
        # TODO: after the loop, return `int(value)`.
        pass
    else:
        # TODO: initialise `value` to +infinity.
        # TODO: iterate over each legal move, this time placing `human_player`.
        #       * Recurse with `is_maximizing=True`.
        #       * Update `value = min(value, score)`.
        #       * Update `beta = min(beta, value)`.
        #       * If `beta <= alpha` break (alpha cut‑off).
        # TODO: return `int(value)` after the loop.
        pass


# ----------------------------------------------------------------
# ai_move -----------------------------------------------------------
# ----------------------------------------------------------------
def ai_move(board: Board, ai_player: str, human_player: str) -> Move:
    """
    Pick the best move for the AI by trying every legal move,
    evaluating it with `minimax_alpha_beta`, and returning the
    coordinates that give the highest score.
    """
    # TODO: set `best_score` to -infinity and `best_move` to a dummy tuple.
    # TODO: loop over every (r, c) from `get_available_moves(board)`.
    #       * Simulate the AI move: `board[r][c] = ai_player`.
    #       * Call `minimax_alpha_beta(board, 0, -inf, +inf, False,
    #                                 ai_player, human_player)` to get its score.
    #       * Undo the simulated move: `board[r][c] = None`.
    #       * If the returned score is greater than `best_score`,
    #         update `best_score` and store `(r, c)` in `best_move`.
    # TODO: after checking all moves, `return best_move`.
    pass
