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


# ----------------------------------------------------------------
# 5. pretty_print()
# ----------------------------------------------------------------
def pretty_print(board: Board) -> None:
    """
    Display the current board in a nice, human‑readable format.

    Expected output (example for a partially‑filled board):

         X |   | O
        -----------
           | X |
        -----------
         O |   | X

    What you need to do:
    1. Create a helper that turns a cell value into a printable symbol.
       * If the cell is None → return a single space `" "`.
       * Otherwise return the stored mark (`"X"` or `"O"`).
    2. Build a list called `rows` where each entry is a string like
       `"X |   | O"` for one board row.
       * Loop over the three rows (index `r` from 0 to 2).
       * Within the row, join the three symbols with `" | "` as separator.
    3. Define a separator string that draws a horizontal line
       (`"\n-----------\n"`).
    4. Print a blank line, then the rows joined by the separator,
       then another blank line – this gives the visual grid.
    """
    # -------------------------------------------------------------
    # IMPLEMENTATION STARTS HERE
    # -------------------------------------------------------------
    pass  # Replace this line with the code described above
    # -------------------------------------------------------------


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
# ----------------------------------------------------------------
# 6. play_game()   – main loop (already complete, but shown as a skeleton)
# ----------------------------------------------------------------
def play_game() -> None:
    """
    Run the interactive console game: Human (X) vs AI (O).

    The flow of the loop is:
    1. Show the board.
    2. If it is the human's turn → read a number 1‑9, convert it to
       row/col, validate, and call `make_move`.
    3. If it is the AI's turn → call `ai_move` to obtain a (row, col)
       and then `make_move` the AI's piece.
    4. After each move, check for a winner or a draw.
    5. If the game ends, print the final board and a message.
    6. Otherwise, switch the current player and repeat.

    What you need to implement:
    * All the logic that is described above is already written in the
      original version – you only need to **uncomment** it (or copy it) into
      this function.
    * Make sure the function **does not return anything**; it should
      simply run the loop until a break occurs.
    * Keep the user prompts clear and validate input as shown in the
      reference implementation.

    Below is a blank template; fill in the body exactly as the original
    code (or copy‑paste it) so the program runs.
    """
    # -------------------------------------------------------------
    # IMPLEMENTATION STARTS HERE
    # -------------------------------------------------------------

    # variables

    while True:
        pretty_print(board)

        if current == human_player:
            while True:
                try:
                    position = int(input("Please make a move (1-9): "))
                except ValueError:
                    print("Enter a valid number between 1-9")

                values = 1

    # -------------------------------------------------------------


if __name__ == "__main__":
    play_game()
