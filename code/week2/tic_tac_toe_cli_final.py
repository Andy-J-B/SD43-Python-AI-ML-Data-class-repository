# --------------------------------------------------------------
# tic_tac_toe_cli_final.py
# --------------------------------------------------------------
# REFERENCE SOLUTION – Console Tic‑Tac‑Toe with a minimax AI.
# --------------------------------------------------------------

from typing import List, Optional, Tuple
import sys

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
    lines.append([board[0][0], board[1][1], board[2][2]])  # \
    lines.append([board[0][2], board[1][1], board[2][0]])  # /

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


def pretty_print(board: Board) -> None:
    def sym(cell):
        return cell if cell is not None else " "

    rows = []
    for r in range(3):
        rows.append(" | ".join(sym(board[r][c]) for c in range(3)))
    separator = "\n-----------\n"
    print("\n" + separator.join(rows) + "\n")


# Up to here
