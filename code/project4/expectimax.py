# ----------------------------------------------------------------
#  EXPECTIMAX (deterministic AI vs. random opponent) – STUDENT SKELETON
# ----------------------------------------------------------------


def expectimax(
    board: Board,  # current board
    depth: int,  # how deep we are in the recursion tree
    is_maximizing: bool,  # True → AI's turn, False → opponent's turn
    ai_player: str,  # AI symbol (e.g. "X")
    human_player: str,  # Opponent symbol (e.g. "O")
) -> float:
    """
    EXPECTIMAX (deterministic‑player vs. random‑player)

    * AI (maximising) picks the move with the highest expected value.
    * Opponent is assumed to move uniformly at random, so we take the
      average of child scores.

    Return a float on the same scale as minimax (+10 for AI win,
    –10 for human win, 0 for draw/unfinished).  Because we average the
    values the result may be non‑integer.
    """
    # ------------------------------------------------------------
    # 1️⃣  Terminal state handling
    # ------------------------------------------------------------
    # TODO: call `check_winner(board)`.  If the winner is `ai_player`
    #       return 10 - depth.  If the winner is `human_player` return
    #       depth - 10.  If `is_draw(board)` is True, return 0.

    # ------------------------------------------------------------
    # 2️⃣  Generate the list of possible moves
    # ------------------------------------------------------------
    # TODO: `moves = get_available_moves(board)`

    # ------------------------------------------------------------
    # 3️⃣  Recurse depending on who is to move
    # ------------------------------------------------------------
    if is_maximizing:
        # TODO: initialise `best` to -infinity.
        # TODO: loop over each (r, c) in `moves`:
        #       * place `ai_player` on the board.
        #       * call `expectimax(board, depth+1, False, ai_player, human_player)`
        #         to get the expected value of this branch.
        #       * undo the move.
        #       * update `best = max(best, score)`.
        # TODO: after the loop, `return best`.
        pass
    else:
        # TODO: initialise an accumulator `total = 0.0`.
        # TODO: loop over each (r, c) in `moves`:
        #       * place `human_player` on the board.
        #       * call `expectimax(board, depth+1, True, ai_player, human_player)`.
        #       * undo the move.
        #       * add the returned score to `total`.
        # TODO: return the average `total / len(moves)` (guard against division by zero).
        pass


# ----------------------------------------------------------------
# ai_move -----------------------------------------------------------
# ----------------------------------------------------------------
def ai_move(board: Board, ai_player: str, human_player: str) -> Move:
    """
    Choose the AI’s next move using an Expectimax search.
    The AI is the maximizing player; the opponent is assumed to act
    uniformly at random, so the expectimax routine returns the *average*
    score of the opponent’s possible replies.
    """
    # TODO: set `best_score` to -infinity and `best_move` to a dummy tuple.
    # TODO: iterate over every (r, c) from `get_available_moves(board)`:
    #       * Simulate the AI move: `board[r][c] = ai_player`.
    #       * Call `expectimax(board, depth=0, is_maximizing=False,
    #                         ai_player=ai_player, human_player=human_player)`
    #         and store the returned score.
    #       * Undo the simulated move: `board[r][c] = None`.
    #       * If this score > `best_score`, update `best_score` and `best_move`.
    # TODO: after the loop, optionally raise an error if no move was found
    #       (should never happen on a non‑full board).
    # TODO: `return best_move`.
    pass
