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
Move = Tuple[int, int]  # row, col (0‑based)


# ----------------------------------------------------------------
# 1. initialise_board()
# ----------------------------------------------------------------
def initialize_board() -> Board:
    """
    Create a brand‑new 3×3 board where every cell is empty.

    What you need to do:
    * Return a list‑of‑lists (`[[...], [...], [...]]`) with three rows.
    * Each inner list must contain three `None` values.
    * No other data (no strings, no numbers) should be placed yet.

    Example return value:
        [[None, None, None],
         [None, None, None],
         [None, None, None]]
    """
    # ---------------------  INSERT YOUR CODE  ---------------------
    pass
    # -------------------------------------------------------------


# ----------------------------------------------------------------
# 2. make_move()
# ----------------------------------------------------------------
def make_move(board: Board, row: int, col: int, player: str) -> bool:
    """
    Try to place `player` ("X" or "O") on the board at the given coordinates.

    Parameters
    ----------
    board : Board
        The current game board (will be mutated in‑place).
    row, col : int
        Zero‑based indices (0 ≤ row < 3, 0 ≤ col < 3).
    player : str
        Either "X" or "O".

    Returns
    -------
    bool
        *True*  – if the cell was empty and the move was applied.
        *False* – if the cell already contained a mark (illegal move).

    What you need to do:
    1. Check whether `board[row][col]` is `None`.
    2. If it is, assign `player` to that cell and return `True`.
    3. Otherwise, leave the board unchanged and return `False`.
    """
    # ---------------------  INSERT YOUR CODE  ---------------------
    pass
    # -------------------------------------------------------------


# ----------------------------------------------------------------
# 3. check_winner()
# ----------------------------------------------------------------
def check_winner(board: Board) -> Optional[str]:
    """
    Examine the board and decide whether either player has won.

    Returns
    -------
    "X" or "O"   – the symbol of the player with three in a row.
    None         – if no player has a winning line yet.

    What you need to do:
    1. Build a collection (`lines`) that contains every possible line
       that can produce a win:
        * 3 rows
        * 3 columns
        * 2 diagonals
    2. Iterate over each line and check:
        * The first cell of the line is **not** `None`.
        * All three cells are equal (`line[0] == line[1] == line[2]`).
    3. If a winning line is found, return the symbol stored in that line
       (`line[0]`).  Otherwise return `None`.
    """
    # ---------------------  INSERT YOUR CODE  ---------------------
    pass
    # -------------------------------------------------------------


# ----------------------------------------------------------------
# 4. is_draw()
# ----------------------------------------------------------------
def is_draw(board: Board) -> bool:
    """
    Determine whether the game has ended in a draw.

    A draw occurs when:
        * Every cell on the board is filled (no `None` left), **and**
        * `check_winner(board)` reports no winner.

    What you need to do:
    1. Verify that *all* cells are not `None`.  This can be done with a
       double `for` loop, a list comprehension, or `all(...)`.
    2. Call `check_winner(board)`.  If it returns `None` **and** the board
       is full, return `True`.  Otherwise return `False`.
    """
    # ---------------------  INSERT YOUR CODE  ---------------------
    pass
    # -------------------------------------------------------------


# ----------------------------------------------------------------
# 5. get_available_moves()
# ----------------------------------------------------------------
def get_available_moves(board: Board) -> List[Move]:
    """
    Return a list of every empty square on the board.

    The result must be a list of `(row, col)` tuples where the cell’s
    value is `None`.  Example for an empty board:
        [(0,0), (0,1), (0,2), (1,0), … , (2,2)]

    What you need to do:
    1. Iterate over rows (`r` from 0‑2) and columns (`c` from 0‑2).
    2. If `board[r][c]` is `None`, append `(r, c)` to a list.
    3. Return the populated list (it may be empty if the board is full).
    """
    # ---------------------  INSERT YOUR CODE  ---------------------
    pass
    # -------------------------------------------------------------


def pretty_print(board: Board) -> None:
    """Print the board in a human‑readable format."""

    def sym(cell):
        return cell if cell is not None else " "

    rows = []
    for r in range(3):
        rows.append(" | ".join(sym(board[r][c]) for c in range(3)))
    separator = "\n-----------\n"
    print("\n" + separator.join(rows) + "\n")


# ----------------------------------------------------------------
# AI LOGIC – **YOU MUST IMPLEMENT THIS FUNCTION**.
# ----------------------------------------------------------------
def ai_move(board: Board, ai_player: str, human_player: str) -> Move:
    """
    Choose the next move for the AI.

    Parameters
    ----------
    board : Board
        Current board state (do NOT modify it directly here; work on a copy if needed).
    ai_player : str
        The symbol the AI plays ('X' or 'O').
    human_player : str
        The symbol the human opponent plays.

    Returns
    -------
    Move
        A tuple (row, col) that is a legal move (must be one of the results
        of `get_available_moves(board)`).

    --------------------------------------------------------------------
    What you should implement:
    • A *very simple* AI such as picking a random legal move is enough for the
      first version.
    • Later you can upgrade it to a minimax algorithm (the final file shows that
      version).
    • Do **not** call `make_move` inside this function – just return the
      coordinates. The game loop will place the piece.
    --------------------------------------------------------------------
    """
    # ----------  INSERT YOUR CODE BELOW  ----------

    # ----------  END OF YOUR CODE  -----------------


# ----------------------------------------------------------------
# MAIN GAME LOOP (already complete)
# ----------------------------------------------------------------
def play_game() -> None:
    """
    Console‑based two‑player game: Human (X) vs AI (O).
    The AI uses `ai_move()` – the student implements that.
    """
    board = initialize_board()
    human_player = "X"
    ai_player = "O"
    current = human_player

    while True:
        pretty_print(board)

        if current == human_player:
            # -------- HUMAN TURN ----------
            while True:
                try:
                    pos = int(input("Your move (1‑9): "))
                except ValueError:
                    print("Please enter a number between 1 and 9.")
                    continue

                if pos < 1 or pos > 9:
                    print("Number out of range.")
                    continue

                row, col = divmod(pos - 1, 3)
                if make_move(board, row, col, human_player):
                    break
                else:
                    print("Square already taken – try again.")
        else:
            # -------- AI TURN ----------
            row, col = ai_move(board, ai_player, human_player)
            make_move(board, row, col, ai_player)
            print(f"AI places {ai_player} at position {row * 3 + col + 1}")

        # -------- CHECK END OF GAME ----------
        winner = check_winner(board)
        if winner:
            pretty_print(board)
            print(f"Game over – {winner} wins!")
            break
        if is_draw(board):
            pretty_print(board)
            print("Game over – it's a draw!")
            break

        # Switch player
        current = ai_player if current == human_player else human_player


if __name__ == "__main__":
    play_game()
