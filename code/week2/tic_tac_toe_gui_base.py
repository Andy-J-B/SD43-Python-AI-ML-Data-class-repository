# --------------------------------------------------------------
# tic_tac_toe_gui_base.py
# --------------------------------------------------------------
# STUDENT STARTER FILE – YOU WILL FILL IN THE AI LOGIC.
# --------------------------------------------------------------

import tkinter as tk
from typing import List, Optional, Tuple

# ----------------------------------------------------------------
#  TYPE ALIASES
# ----------------------------------------------------------------
Board = List[List[Optional[str]]]  # 3x3 grid containing "X", "O", or None
Move = Tuple[int, int]  # row, col indices (0‑based)


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
#  AI LOGIC – **YOU MUST IMPLEMENT THIS FUNCTION**.
# ----------------------------------------------------------------
def ai_move(board: Board, ai_player: str, human_player: str) -> Move:
    """
    Choose the next move for the AI.

    Parameters
    ----------
    board : Board
        Current game state (do NOT modify it directly here – work on a copy).
    ai_player : str
        The symbol the AI plays ('X' or 'O').
    human_player : str
        The symbol the human opponent plays.

    Returns
    -------
    Move
        A tuple (row, col) indicating where the AI wants to play.
        The returned move **must** be one of the positions returned by
        `get_available_moves(board)`.

    --------------------------------------------------------------------
    What you should implement:
    • A *very simple* AI (random choice) is acceptable for a first version.
      The instructor can later ask the student to upgrade it to a minimax
      algorithm (the final file shows that version).
    • Do **not** call `make_move` inside this function; just return the
      chosen coordinates. The game controller will place the mark.
    --------------------------------------------------------------------
    """
    # ----------  INSERT YOUR CODE BELOW  ----------

    # ----------  END OF YOUR CODE  -----------------


# ----------------------------------------------------------------
#  TKINTER UI (fully functional – no changes required)
# ----------------------------------------------------------------
class TicTacToeGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Tic‑Tac‑Toe – Human vs Minimax AI")

        self.board = initialize_board()
        self.buttons: List[List[tk.Button]] = []

        self.human_player = "X"
        self.ai_player = "O"
        self.current_player = self.human_player  # human starts

        # --- MODIFICATION START ---
        # This 1x1 pixel image is a trick to allow us to
        # specify button size in pixels (width/height)
        # instead of text units.
        self.pixel = tk.PhotoImage(width=1, height=1)
        # --- MODIFICATION END ---

        self.status_label = tk.Label(
            self.root, text="Your turn (X)", font=("Arial", 14)
        )
        self.status_label.grid(row=0, column=0, columnspan=3, pady=5)

        # --- MODIFICATION START ---
        # Define pixel size and font size for the buttons
        button_pixel_size = 100  # 100x100 pixels
        font_size = 40  # Larger font to fill the button
        # --- MODIFICATION END ---

        for r in range(3):
            row_btns = []
            for c in range(3):
                # --- MODIFICATION START ---
                # The Button creation is updated to use pixel sizing
                btn = tk.Button(
                    self.root,
                    text=" ",
                    font=("Arial", font_size, "bold"),  # Made font bigger & bold
                    image=self.pixel,  # Use the 1x1 pixel image
                    width=button_pixel_size,  # Set width in pixels
                    height=button_pixel_size,  # Set height in pixels
                    compound="center",  # Display text *over* the image
                    command=lambda row=r, col=c: self.on_human_click(row, col),
                )
                # --- MODIFICATION END ---
                btn.grid(row=r + 1, column=c, padx=5, pady=5)
                row_btns.append(btn)
            self.buttons.append(row_btns)

    # ----------------------------------------------------------------
    def on_human_click(self, row: int, col: int) -> None:
        """Handle a click from the human player."""
        if self.current_player != self.human_player:
            return  # ignore clicks when it's AI's turn

        if not make_move(self.board, row, col, self.human_player):
            self.status_label.config(text="Square already taken – try another")
            return

        self.update_ui()
        if self.end_of_game_check():
            return

        # Switch to AI
        self.current_player = self.ai_player
        self.root.after(200, self.perform_ai_move)  # tiny delay for UI responsiveness

    # ----------------------------------------------------------------
    def perform_ai_move(self) -> None:
        """Ask the AI for a move and execute it."""
        row, col = ai_move(self.board, self.ai_player, self.human_player)
        make_move(self.board, row, col, self.ai_player)
        self.update_ui()
        if self.end_of_game_check():
            return
        # Switch back to human
        self.current_player = self.human_player
        self.status_label.config(text="Your turn (X)")

    # ----------------------------------------------------------------
    def update_ui(self) -> None:
        """Refresh all button labels to match `self.board`."""
        for r in range(3):
            for c in range(3):
                value = self.board[r][c]
                self.buttons[r][c].config(text=value if value else " ")

    # ----------------------------------------------------------------
    def end_of_game_check(self) -> bool:
        """Check for win / draw. Display a message and disable the board."""
        winner = check_winner(self.board)
        if winner:
            self.status_label.config(text=f"Game over – {winner} wins!")
            self.disable_all_buttons()
            return True
        if is_draw(self.board):
            self.status_label.config(text="Game over – it's a draw!")
            self.disable_all_buttons()
            return True
        return False

    # ----------------------------------------------------------------
    def disable_all_buttons(self) -> None:
        for row in self.buttons:
            for btn in row:
                btn.config(state=tk.DISABLED)


# ----------------------------------------------------------------
def main() -> None:
    root = tk.Tk()
    TicTacToeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
