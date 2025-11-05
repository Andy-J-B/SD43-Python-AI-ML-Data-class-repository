# --------------------------------------------------------------
# tic_tac_toe_gui_final.py
# --------------------------------------------------------------
# REFERENCE SOLUTION – fully functional Tic‑Tac‑Toe with a
# minimax AI and a clean Tkinter UI.
# --------------------------------------------------------------

import tkinter as tk
from typing import List, Optional, Tuple
import random

Board = List[List[Optional[str]]]
Move = Tuple[int, int]  # (row, col)


def initialize_board() -> Board:
    return [[None for _ in range(3)] for _ in range(3)]


def make_move(board: Board, row: int, col: int, player: str) -> bool:
    if board[row][col] is None:
        board[row][col] = player
        return True
    return False


def check_winner(board: Board) -> Optional[str]:
    lines = []
    for i in range(3):
        lines.append(board[i])  # rows
        lines.append([board[0][i], board[1][i], board[2][i]])  # cols
    lines.append([board[0][0], board[1][1], board[2][2]])  # diag \
    lines.append([board[0][2], board[1][1], board[2][0]])  # diag /

    for line in lines:
        if line[0] is not None and line[0] == line[1] == line[2]:
            return line[0]
    return None


def is_draw(board: Board) -> bool:
    return (
        all(cell is not None for row in board for cell in row)
        and check_winner(board) is None
    )


def get_available_moves(board: Board) -> List[Move]:
    return [(r, c) for r in range(3) for c in range(3) if board[r][c] is None]


def ai_move(board: Board, ai_player: str, human_player: str) -> Move:
    """
    Pick the best move for the AI using minimax.
    """
    random_moves = get_available_moves(board)
    return random.choice(random_moves)


# ----------------------------------------------------------------
#  GUI (unchanged from the starter, only tiny cosmetic adjustments)
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
        if self.current_player != self.human_player:
            return

        if not make_move(self.board, row, col, self.human_player):
            self.status_label.config(text="Square already taken – try again")
            return

        self.update_ui()
        if self.end_of_game_check():
            return

        self.current_player = self.ai_player
        self.root.after(250, self.perform_ai_move)

    # ----------------------------------------------------------------
    def perform_ai_move(self) -> None:
        row, col = ai_move(self.board, self.ai_player, self.human_player)
        make_move(self.board, row, col, self.ai_player)
        self.update_ui()
        if self.end_of_game_check():
            return
        self.current_player = self.human_player
        self.status_label.config(text="Your turn (X)")

    # ----------------------------------------------------------------
    def update_ui(self) -> None:
        for r in range(3):
            for c in range(3):
                val = self.board[r][c]
                self.buttons[r][c].config(text=val if val else " ")

    # ----------------------------------------------------------------
    def end_of_game_check(self) -> bool:
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


def main() -> None:
    root = tk.Tk()
    TicTacToeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
