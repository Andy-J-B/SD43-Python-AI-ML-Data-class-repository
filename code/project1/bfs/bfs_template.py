# ------------------------------------------------------------
# bfs_template.py
# ------------------------------------------------------------
# This is the starter file you will receive at the beginning of
# the lesson.  DO NOT rename any functions or change their
# signatures – the test script expects exactly these names.
#
# Your task is to fill in the bodies marked with TODO.
# ------------------------------------------------------------

from collections import deque
from typing import List, Tuple, Optional

# ----------------------------------------------------------------------
# Helper data type: a position in the grid is represented as (row, col)
# ----------------------------------------------------------------------
Position = Tuple[int, int]


def bfs(
    grid: List[List[int]], start: Position, goal: Position
) -> Optional[List[Position]]:
    """
    Perform a Breadth‑First Search on a 2‑D grid.

    Parameters
    ----------
    grid  : List[List[int]]
        A matrix where 0 = free cell, 1 = obstacle/wall.
    start : Position
        (row, col) coordinates of the start cell.
    goal  : Position
        (row, col) coordinates of the target cell.

    Returns
    -------
    path : List[Position] | None
        The shortest path from start to goal as a list of positions,
        INCLUDING start and goal.  Return None if no path exists.
    """
    # ------------------------------------------------------------------
    # TODO 1: Validate inputs (grid non‑empty, start/goal inside grid,
    #          cells are not walls).  If anything is invalid, raise
    #          ValueError with a helpful message.
    # ------------------------------------------------------------------
    # YOUR CODE HERE ---------------------------------------------------

    # ------------------------------------------------------------------
    # TODO 2: Implement the BFS algorithm.
    #   - Use a queue (collections.deque) for the frontier.
    #   - Keep a dictionary `came_from` that maps a node -> predecessor.
    #   - Explore the four orthogonal neighbours (up, down, left, right)
    #     that are inside the grid and not walls.
    #   - Stop when you pop the goal node.
    # ------------------------------------------------------------------
    # YOUR CODE HERE ---------------------------------------------------

    # ------------------------------------------------------------------
    # TODO 3: Reconstruct the path from goal back to start using
    #         the `came_from` dictionary and return it as a list.
    #         Remember to reverse it so it goes start → goal.
    # ------------------------------------------------------------------
    # YOUR CODE HERE ---------------------------------------------------

    # If the goal was never reached, return None
    return None


grid1 = [
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 0, 0, 1],
    [1, 1, 0, 0],
]
start1 = (0, 0)
goal1 = (3, 3)
expected1 = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (3, 2), (3, 3)]

print(bfs(grid1, start1, goal1))
print(bfs(grid1, start1, goal1) == expected1)

# Will print true if passed test case, false if not.
