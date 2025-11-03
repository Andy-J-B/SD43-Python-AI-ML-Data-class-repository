# 2048_base.py
# ------------------------------------------------------------
# Starter code for the “Expectimax 2048” assignment.
# ------------------------------------------------------------
# All functions that the test suite (or the CLI) will call are
# present, but every body is a stub that raises NotImplementedError.
# The student’s job is to replace each stub with a correct
# implementation while keeping the function signatures unchanged.
#
# ------------------- HOW TO USE -------------------
# 1. Open this file in your favourite editor / IDE.
# 2. Read the doc‑string of each function – it tells you exactly
#    what the function must do.
# 3. Implement the logic – you may copy ideas from the lecture,
#    from the solution file, or from any reference implementation,
#    but the final code must *only* use explicit loops / if‑statements
#    (no list‑comprehensions, no one‑liner dict look‑ups).
# 4. Run the supplied test script (`test_2048_cli.py`) after each
#    milestone to see red/green feedback.
# 5. When all tests pass, run this file directly (`python 2048_base.py`);
#    the UI will work and you can watch the AI play.
# ------------------------------------------------------------

import random
from copy import deepcopy
from typing import List, Tuple

SIZE = 4  # board is 4 × 4
START_TILES = 2  # how many random tiles are placed at start


# ----------------------------------------------------------------------
# 1️⃣  BOARD HELPERS – WRITE THESE WITH EXPLICIT LOOPS
# ----------------------------------------------------------------------
def empty_cells(board: List[List[int]]) -> List[Tuple[int, int]]:
    """
    Return a list of (row, col) positions that are empty (contain 0).

    REQUIRED IMPLEMENTATION
    -----------------------
    Use two nested while‑loops (or for‑loops) to scan the board.
    Whenever you find a 0, append the tuple (row, col) to a list and
    finally return that list.
    """
    raise NotImplementedError


def add_random_tile(board: List[List[int]]) -> None:
    """
    Place a 2 (90 % chance) or a 4 (10 % chance) on a random empty cell.
    The board must be mutated in place.

    REQUIRED IMPLEMENTATION
    -----------------------
    1. Call `empty_cells(board)` to obtain a list of free positions.
    2. Pick one of them at random (`random.randrange` or `random.choice`).
    3. With probability 0.1 write a 4, otherwise write a 2.
    """
    raise NotImplementedError


def init_board() -> List[List[int]]:
    """
    Create a fresh board with START_TILES random tiles.
    This function is imported by the test harness – DO NOT rename.
    """
    raise NotImplementedError


# ----------------------------------------------------------------------
# 2️⃣  MOVE LOGIC – WRITE THESE WITH EXPLICIT LOOPS
# ----------------------------------------------------------------------
def compress(row: List[int]) -> List[int]:
    """
    Slide all non‑zero numbers in *row* to the left, preserving order.
    Example: [2,0,2,4] → [2,2,4,0]

    REQUIRED IMPLEMENTATION
    -----------------------
    * Build a new list containing only the non‑zero elements (use a while‑loop).
    * Append 0s until the list length becomes SIZE.
    """
    raise NotImplementedError


def merge(row: List[int]) -> Tuple[List[int], int]:
    """
    Merge equal neighbours from left to right.
    Returns (new_row, points_gained).

    Example: [2,2,4,0] → ([4,0,4,0], 4)

    REQUIRED IMPLEMENTATION
    -----------------------
    * Walk through the row with an index i from 0 to SIZE‑2.
    * If `row[i]` is non‑zero and equals `row[i+1]`,
      double `row[i]`, set `row[i+1]` to 0 and add the new value
      to the points counter.
    * Return the (now‑modified) row and the points earned.
    """
    raise NotImplementedError


def move_left(board: List[List[int]]) -> Tuple[List[List[int]], int, bool]:
    """
    Execute a left move.
    Returns (new_board, points_gained, moved_flag).

    REQUIRED IMPLEMENTATION
    -----------------------
    * For each row of the board:
          1) compress,
          2) merge (collect the points),
          3) compress again.
    * If any row changed, set `moved` to True.
    * Return the new board, the total points and the moved flag.
    """
    raise NotImplementedError


def rotate(board: List[List[int]]) -> List[List[int]]:
    """
    Rotate the board 90° clockwise.
    Implement with explicit loops (no zip / list‑comprehensions).

    REQUIRED IMPLEMENTATION
    -----------------------
    * Create a new empty 4×4 matrix.
    * For each cell (r, c) in the original board write its value
      into position (c, SIZE‑1‑r) of the new matrix.
    * Return the new matrix.
    """
    raise NotImplementedError


def move(board: List[List[int]], direction: str) -> Tuple[List[List[int]], int, bool]:
    """
    Perform a move in the given direction.
    direction must be one of: 'Up', 'Down', 'Left', 'Right'.

    The rotation mapping that the UI expects is:
        Left  → 0 rotations
        Down  → 1 rotation
        Right → 2 rotations
        Up    → 3 rotations

    REQUIRED IMPLEMENTATION
    -----------------------
    1. Translate `direction` into the number of clockwise rotations
       (use a series of if/elif statements – **do not use a dict
       literal**).
    2. Rotate the board that many times.
    3. Call `move_left` on the rotated board.
    4. Rotate the result back to the original orientation.
    5. Return the final board, the points earned and the moved flag.
    """
    raise NotImplementedError


def can_move(board: List[List[int]]) -> bool:
    """
    Return True if at least one legal move exists (including the chance
    of inserting a new random tile).

    REQUIRED IMPLEMENTATION
    -----------------------
    * If `empty_cells(board)` is non‑empty → return True.
    * Otherwise try the four directions; if any call to `move`
      returns `moved == True` then a move is possible.
    * If none of the above, return False.
    """
    raise NotImplementedError


# ----------------------------------------------------------------------
# 3️⃣  EXPECTIMAX – STUDENT IMPLEMENTATION REQUIRED
# ----------------------------------------------------------------------
def evaluation(board: List[List[int]]) -> float:
    """
    Heuristic required by the specification:
        sum_of_tiles + 1000 * number_of_empty_cells
    Must return a float.

    IMPLEMENTATION NOTE
    -------------------
    * Compute the sum of all values on the board (explicit loops).
    * Call `empty_cells(board)` to obtain the number of empty cells.
    * Return `float(total + 1000 * empty)`.
    """
    raise NotImplementedError


def expectimax(board: List[List[int]], depth: int, player: bool) -> float:
    """
    Expectimax recursion.
    * player == True  → max node (AI turn)
    * player == False → chance node (random tile appears)

    TERMINATION
    ------------
    * If `depth == 0` **or** the game is over (`can_move(board)` is False)
      return `evaluation(board)`.

    MAX NODE (player == True)
    -------------------------
    * For every legal direction, obtain the child board with `move`.
    * Recurse with `depth‑1` and `player=False`.
    * Return the maximum child value (ignore directions that do not move).

    CHANCE NODE (player == False)
    -----------------------------
    * For each empty cell generate two children:
          – tile 2 with probability 0.9
          – tile 4 with probability 0.1
    * Recurse on each child with `depth‑1` and `player=True`.
    * Return the probability‑weighted average of all child values.
    """
    raise NotImplementedError


def get_best_move(board: List[List[int]], depth: int = 3) -> str:
    """
    Return the direction ('Up','Down','Left','Right') that yields the
    highest Expectimax value (depth‑1, player=False) among all legal moves.

    IMPLEMENTATION NOTE
    -------------------
    * Evaluate the four directions exactly as `move` does.
    * Keep track of the best value and the corresponding direction.
    * Return the direction with the highest value.
    """
    raise NotImplementedError


def ai_move(
    board: List[List[int]], depth: int = 3
) -> Tuple[List[List[int]], int, bool]:
    """
    Helper used by the CLI – performs ONE AI move.
    Returns (new_board, points_gained, moved_flag) exactly as `move`
    does.

    IMPLEMENTATION NOTE
    -------------------
    * Call `get_best_move(board, depth)` to obtain the direction.
    * Return the result of `move(board, direction)`.
    """
    raise NotImplementedError


# ------------------------------------------------------------
# 4️⃣  SIMPLE TEXT USER INTERFACE – STUDENT IMPLEMENTATION REQUIRED
# ------------------------------------------------------------
def pretty_print(board: List[List[int]]) -> None:
    """
    Print the current board to the terminal in a compact grid.

    REQUIRED IMPLEMENTATION
    -----------------------
    1. Build the horizontal separator line:  "+------" repeated SIZE times
       followed by a final "+" (e.g. for a 4×4 board:
       "+------+------+------+------+").
    2. Loop over the rows (use a while‑loop, **no list‑comprehensions**).
       For each row:
           • Build a string that contains the 4 cell values separated by "|".
             - If a cell value is 0, display an empty string.
             - Each cell must be **center‑aligned** and occupy exactly 6 characters.
               You can achieve this with the format specifier `f"{val or '':^6}"`.
           • Print the line that begins and ends with "|" (the row itself).
           • Print the separator line again.
    3. The function returns `None`; it only prints to stdout.

    """
    raise NotImplementedError


def cli_game() -> None:
    """
    Main loop for the command‑line version of 2048.
    The student must implement the complete interactive loop that:
        • Creates a fresh board with `init_board()`.
        • Keeps track of the current score.
        • Re‑displays the board each turn (using `pretty_print`).
        • Checks `can_move(board)` – if no move is possible, announces
          “GAME OVER” and exits the loop.
        • Reads a single‑character command from the user:
            w → "Up"
            a → "Left"
            s → "Down"
            d → "Right"
            i → let the AI make a move
            q → quit the program
          Any other character should cause a brief “Invalid key – try again.”
          message and continue the loop.
        • For a human move:
            – call `move(board, direction)` and obtain `(new_board, pts, moved)`.
        • For an AI move:
            – call `ai_move(board, depth=3)` (you may keep the depth hard‑coded).
        • If `moved` is True:
            – replace `board` with `new_board`,
            – add `pts` to the running `score`,
            – call `add_random_tile(board)` to insert a new tile.
          Else print a short warning (“Move didn't change the board …”).
        • After each successful turn, print the updated score
          (`print(f"Score: {score}")`).

    """
    raise NotImplementedError


if __name__ == "__main__":
    cli_game()
