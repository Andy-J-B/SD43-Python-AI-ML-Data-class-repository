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
#  MINIMAX WITH ALPHA‑BETA PRUNING
# ----------------------------------------------------------------


def minimax_alpha_beta(
    board: Board,  # current game board (list of lists)
    depth: int,  # how many moves deep we are in the tree
    alpha: float,  # best value that the maximiser can guarantee
    beta: float,  # best value that the minimiser can guarantee
    is_maximizing: bool,  # True → AI's turn, False → human's turn
    ai_player: str,  # symbol used by the AI (e.g. "X")
    human_player: str,  # symbol used by the human (e.g. "O")
) -> int:
    """
    MINIMAX WITH ALPHA‑BETA PRUNING

    Returns an integer score (+10 for AI win, –10 for Human win, 0 for draw)
    while cutting off branches that cannot influence the final decision.
    """
    # ------------------------------------------------------------
    # 1️⃣  Terminal state check (win / loss / draw)
    # ------------------------------------------------------------
    winner = check_winner(board)  # does anyone already have three in a row?
    if winner == ai_player:  # AI has already won
        return 10 - depth  # sooner wins are better
    if winner == human_player:  # Human has already won
        return depth - 10  # later losses are slightly less bad
    if is_draw(board):  # board full with no winner
        return 0  # neutral outcome

    # ------------------------------------------------------------
    # 2️⃣  Recursive exploration with cut‑offs
    # ------------------------------------------------------------
    if is_maximizing:  # ----- AI's move (we want the highest value)
        value = -float("inf")  # start lower than any possible score
        for r, c in get_available_moves(board):  # iterate over every empty cell
            board[r][c] = ai_player  # make a tentative AI move
            # recurse: now it will be the human's turn (is_maximizing=False)
            score = minimax_alpha_beta(
                board, depth + 1, alpha, beta, False, ai_player, human_player
            )
            board[r][c] = None  # undo the tentative move (backtrack)

            value = max(value, score)  # keep the best score we have seen
            alpha = max(alpha, value)  # update α (best guarantee for maximiser)

            if alpha >= beta:  # α‑β cut‑off → this branch cannot improve outcome
                break  # stop exploring further children
        return int(value)  # cast to int for consistency with original API

    else:  # ----- Human's move (we want the lowest value)
        value = float("inf")  # start higher than any possible score
        for r, c in get_available_moves(board):
            board[r][c] = human_player  # tentative human move
            score = minimax_alpha_beta(
                board, depth + 1, alpha, beta, True, ai_player, human_player
            )
            board[r][c] = None  # undo move

            value = min(
                value, score
            )  # keep the worst score for the AI (best for human)
            beta = min(beta, value)  # update β (best guarantee for minimiser)

            if beta <= alpha:  # β‑α cut‑off → no need to explore further
                break
        return int(value)  # final minimised value for this branch


# ----------------------------------------------------------------
# ai_move -----------------------------------------------------------
# ----------------------------------------------------------------
def ai_move(board: Board, ai_player: str, human_player: str) -> Move:
    """
    Choose the optimal move for the AI using the alpha‑beta minimax.
    """
    best_score = -float("inf")  # initialise with the worst possible score
    best_move: Move = (-1, -1)  # placeholder for the best coordinates

    # ------------------------------------------------------------
    # Try every legal move, evaluate it with alpha‑beta, keep the best.
    # ------------------------------------------------------------
    for r, c in get_available_moves(board):
        board[r][c] = ai_player  # simulate AI playing here
        score = minimax_alpha_beta(
            board, 0, -float("inf"), float("inf"), False, ai_player, human_player
        )
        board[r][c] = None  # undo the simulation

        if score > best_score:  # found a better move?
            best_score = score
            best_move = (r, c)  # remember its coordinates

    return best_move  # return the optimal cell for the AI


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
