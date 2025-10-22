# ------------------------------------------------------------
# dfs_template.py
# ------------------------------------------------------------
# Starter file for the DFS project.
# DO NOT change the name or signature of the function `dfs`.
# Fill in the bodies marked with TODO.
# ------------------------------------------------------------

from typing import List, Tuple, Optional, Set

# A grid cell is identified by (row, col)
Position = Tuple[int, int]


def dfs(
    grid: List[List[int]], start: Position, goal: Position
) -> Optional[List[Position]]:
    """
    Depth‑First Search on a 2‑D grid (4‑directional moves).

    Returns the **first** path found from `start` to `goal`
    (not guaranteed to be shortest).  If no path exists, returns None.
    """
    # ------------------------------------------------------------------
    # TODO 1 – Input validation (same checks as in BFS)
    # ------------------------------------------------------------------
    # YOUR CODE HERE ---------------------------------------------------

    # ------------------------------------------------------------------
    # TODO 2 – Implement DFS using an explicit stack.
    #           Keep a dictionary `came_from` mapping each visited node
    #           to its predecessor so we can rebuild the path.
    #           Stop as soon as the goal is popped from the stack.
    # ------------------------------------------------------------------
    # YOUR CODE HERE ---------------------------------------------------

    # ------------------------------------------------------------------
    # TODO 3 – Reconstruct the path from goal back to start (as in BFS)
    # ------------------------------------------------------------------
    # YOUR CODE HERE ---------------------------------------------------

    # If the goal was never reached:
    return None
