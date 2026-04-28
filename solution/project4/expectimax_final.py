# --------------------------------------------------------------
# tic_tac_toe_cli_base.py
# --------------------------------------------------------------
# STUDENT STARTER FILE – ONLY THE AI LOGIC IS MISSING.
# --------------------------------------------------------------

import random
from typing import List, Optional, Tuple

# ----------------------------------------------------------------
# TYPE ALIASES
# ----------------------------------------------------------------
Board = List[List[Optional[str]]]  # 3x3 grid containing "X", "O", or None
Move = Tuple[int, int]  # (row, col) – 0‑based indices


# ----------------------------------------------------------------
# 1. initialise_board()
# ----------------------------------------------------------------
def initialize_board() -> Board:
    """
    Create a brand‑new 3×3 board where every cell is empty.
    """
    # ---------------------  INSERT YOUR CODE  ---------------------
    return [[None for _ in range(3)] for _ in range(3)]
    # -------------------------------------------------------------


# ----------------------------------------------------------------
# 2. make_move()
# ----------------------------------------------------------------
def make_move(board: Board, row: int, col: int, player: str) -> bool:
    """
    Try to place `player` ("X" or "O") on the board at the given coordinates.
    """
    # ---------------------  INSERT YOUR CODE  ---------------------
    if board[row][col] is None:
        board[row][col] = player
        return True
    return False
    # -------------------------------------------------------------


# ----------------------------------------------------------------
# 3. check_winner()
# ----------------------------------------------------------------
def check_winner(board: Board) -> Optional[str]:
    """
    Examine the board and decide whether either player has won.
    """
    # ---------------------  INSERT YOUR CODE  ---------------------
    lines = []

    for i in range(3):
        # rows
        lines.append(board[i])
        # columns
        lines.append([board[0][i], board[1][i], board[2][i]])

    # two diagonals
    lines.append([board[0][0], board[1][1], board[2][2]])  # \
    lines.append([board[0][2], board[1][1], board[2][0]])  # /

    for line in lines:
        if line[0] is not None and line[0] == line[1] == line[2]:
            return line[0]

    return None
    # -------------------------------------------------------------


# ----------------------------------------------------------------
# 4. is_draw()
# ----------------------------------------------------------------
def is_draw(board: Board) -> bool:
    """
    Determine whether the game has ended in a draw.
    """
    # ---------------------  INSERT YOUR CODE  ---------------------
    board_full = all(cell is not None for row in board for cell in row)
    return board_full and check_winner(board) is None
    # -------------------------------------------------------------


# ----------------------------------------------------------------
# 5. get_available_moves()
# ----------------------------------------------------------------
def get_available_moves(board: Board) -> List[Move]:
    """
    Return a list of every empty square on the board.
    """
    # ---------------------  INSERT YOUR CODE  ---------------------
    moves: List[Move] = []
    for r in range(3):
        for c in range(3):
            if board[r][c] is None:
                moves.append((r, c))
    return moves
    # -------------------------------------------------------------


# ----------------------------------------------------------------
# 5. pretty_print()
# ----------------------------------------------------------------
def pretty_print(board: Board) -> None:
    """
    Display the current board in a nice, human‑readable format.
    """

    # -------------------------------------------------------------
    # IMPLEMENTATION STARTS HERE
    # -------------------------------------------------------------
    def sym(cell: Optional[str]) -> str:
        """Convert a board cell into a printable character."""
        return cell if cell is not None else " "

    rows: List[str] = []
    for r in range(3):
        rows.append(" | ".join(sym(board[r][c]) for c in range(3)))

    separator = "\n-----------\n"
    print("\n" + separator.join(rows) + "\n")
    # -------------------------------------------------------------


# ----------------------------------------------------------------
#  MINIMAX AI
# ----------------------------------------------------------------


# ----------------------------------------------------------------
#  EXPECTIMAX (deterministic AI vs. random opponent)
# ----------------------------------------------------------------


def expectimax(
    board: Board,  # current board (list of lists)
    depth: int,  # recursion depth – how many moves ahead we are
    is_maximizing: bool,  # True → AI's turn (maximiser), False → opponent's turn
    ai_player: str,  # symbol used by the AI (e.g. "X")
    human_player: str,  # symbol used by the random opponent (e.g. "O")
) -> float:
    """
    EXPECTIMAX (deterministic‑player vs. random‑player)

    * AI (maximising) chooses the move with the highest expected value.
    * The opponent is assumed to play uniformly at random, so we take the
      *average* of the child‑node scores instead of the minimum.

    Returns a float on the same scale as minimax (+10 for AI win,
    –10 for human win, 0 for draw/unfinished).  Because we average the
    values the result may be non‑integer.
    """
    # ------------------------------------------------------------
    # 1️⃣  Check for terminal states (win / loss / draw)
    # ------------------------------------------------------------
    winner = check_winner(board)  # does someone already have three in a row?
    if winner == ai_player:  # AI already won
        return 10 - depth  # quicker wins are better
    if winner == human_player:  # opponent already won
        return depth - 10  # slower losses are slightly less bad
    if is_draw(board):  # board full without a winner
        return 0  # neutral outcome

    # ------------------------------------------------------------
    # 2️⃣  Gather all possible moves from the current board
    # ------------------------------------------------------------
    moves = get_available_moves(board)  # list of (row, col) tuples that are empty

    # ------------------------------------------------------------
    # 3️⃣  Recurse depending on whose turn it is
    # ------------------------------------------------------------
    if is_maximizing:  # ----- AI's turn (we want the highest expected value)
        best = -float("inf")  # start lower than any possible score
        for r, c in moves:  # examine each legal move
            board[r][c] = ai_player  # simulate AI playing here
            score = expectimax(
                board, depth + 1, False, ai_player, human_player
            )  # opponent's turn in the recursive call
            board[r][c] = None  # undo the simulation (backtrack)
            best = max(best, score)  # keep the best expected score found
        return best  # return the maximal expected value

    else:  # ----- Opponent's turn (random player)
        total = 0.0  # accumulator for the sum of child scores
        for r, c in moves:
            board[r][c] = human_player  # simulate opponent playing here
            score = expectimax(
                board, depth + 1, True, ai_player, human_player
            )  # back to AI's turn
            board[r][c] = None  # undo move
            total += score  # add this branch's value
        # Expected value = average of all possible child scores.
        # Guard against division by zero (should never happen because moves is non‑empty).
        return total / len(moves) if moves else 0.0


# ----------------------------------------------------------------
# ai_move -----------------------------------------------------------
# ----------------------------------------------------------------
def ai_move(board: Board, ai_player: str, human_player: str) -> Move:
    """
    Choose the AI’s next move using an Expectimax search.
    The AI is the maximising player; the opponent is assumed to act
    uniformly at random, so the expectimax routine returns the *average*
    score of the opponent’s possible replies.
    """
    best_score = -float("inf")  # initialise with the worst possible score
    best_move: Move = (-1, -1)  # placeholder for the best coordinates

    # ------------------------------------------------------------
    # Examine every empty square, evaluate it with expectimax,
    # keep the move that yields the highest expected utility.
    # ------------------------------------------------------------
    for r, c in get_available_moves(board):
        board[r][c] = ai_player  # simulate AI playing this move
        # After the AI’s move it is the opponent’s turn, so is_maximizing=False.
        score = expectimax(
            board,
            depth=0,
            is_maximizing=False,
            ai_player=ai_player,
            human_player=human_player,
        )
        board[r][c] = None  # undo the simulation

        if score > best_score:  # found a better expected value?
            best_score = score
            best_move = (r, c)  # remember its location

    # ------------------------------------------------------------
    # Safety check – in a legal game there is always at least one move.
    # ------------------------------------------------------------
    if best_move == (-1, -1):
        raise RuntimeError("ai_move called on a full board")

    return best_move  # return the coordinates of the optimal move


# ----------------------------------------------------------------
# MAIN GAME LOOP (already complete)
# ----------------------------------------------------------------
def play_game() -> None:
    """
    Run the interactive console game: Human (X) vs AI (O).
    """
    # -------------------------------------------------------------
    # IMPLEMENTATION STARTS HERE
    # -------------------------------------------------------------
    board = initialize_board()
    human_player = "X"
    ai_player = "O"
    current_player = human_player  # Human starts

    while True:
        pretty_print(board)

        if current_player == human_player:
            # -------- Human turn ----------
            while True:
                try:
                    move = input("Enter your move (1‑9): ").strip()
                    pos = int(move)
                    if not 1 <= pos <= 9:
                        raise ValueError
                    row, col = divmod(pos - 1, 3)
                    if make_move(board, row, col, human_player):
                        break
                    else:
                        print("That square is already taken. Try again.")
                except ValueError:
                    print(
                        "Please enter a number from 1 to 9 corresponding to an empty square."
                    )
        else:
            # -------- AI turn ----------
            row, col = ai_move(board, ai_player, human_player)
            make_move(board, row, col, ai_player)
            print(f"AI chooses square {row * 3 + col + 1}")

        # -------- Check end of game ----------
        winner = check_winner(board)
        if winner:
            pretty_print(board)
            if winner == human_player:
                print("Congratulations – you win!")
            else:
                print("AI wins – better luck next time!")
            break

        if is_draw(board):
            pretty_print(board)
            print("It's a draw!")
            break

        # -------- Switch player ----------
        current_player = ai_player if current_player == human_player else human_player
    # -------------------------------------------------------------


if __name__ == "__main__":
    play_game()
