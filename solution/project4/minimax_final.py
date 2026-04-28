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


# # ----------------------------------------------------------------
# # AI LOGIC – **YOU MUST IMPLEMENT THIS FUNCTION**.
# # ----------------------------------------------------------------
# def ai_move(board: Board, ai_player: str, human_player: str) -> Move:
#     """
#     Choose the next move for the AI.

#     For this starter version we simply pick a random legal move.
#     """
#     # ----------  INSERT YOUR CODE BELOW  ----------
#     available = get_available_moves(board)
#     # `available` is guaranteed to be non‑empty when this function is called.
#     return random.choice(available)
#     # ----------  END OF YOUR CODE  -----------------


# ----------------------------------------------------------------
#  MINIMAX AI
# ----------------------------------------------------------------
def minimax(
    board: Board, depth: int, is_maximizing: bool, ai_player: str, human_player: str
) -> int:
    winner = check_winner(board)
    if winner == ai_player:
        return 10 - depth  # quicker win is better
    if winner == human_player:
        return depth - 10  # slower loss is better
    if is_draw(board):
        return 0

    if is_maximizing:
        best = -float("inf")
        for r, c in get_available_moves(board):
            board[r][c] = ai_player
            score = minimax(board, depth + 1, False, ai_player, human_player)
            board[r][c] = None
            best = max(best, score)
        return best
    else:
        best = float("inf")
        for r, c in get_available_moves(board):
            board[r][c] = human_player
            score = minimax(board, depth + 1, True, ai_player, human_player)
            board[r][c] = None
            best = min(best, score)
        return best


def ai_move(board: Board, ai_player: str, human_player: str) -> Move:
    """
    Pick the best move for the AI using minimax.
    """
    best_score = -float("inf")
    best_move: Move = (-1, -1)

    for r, c in get_available_moves(board):
        board[r][c] = ai_player
        score = minimax(board, 0, False, ai_player, human_player)
        board[r][c] = None
        if score > best_score:
            best_score = score
            best_move = (r, c)

    return best_move


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
