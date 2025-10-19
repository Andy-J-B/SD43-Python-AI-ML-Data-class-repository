# ------------------------------------------------------------
# ucs_template.py
# ------------------------------------------------------------
# Starter file for the Uniform‑Cost Search (UCS) project.
# DO NOT rename the function `ucs`.
# Fill in the bodies marked with TODO.
# ------------------------------------------------------------

import heapq
from typing import List, Tuple, Optional, Dict

# Position on the grid
Position = Tuple[int, int]


def ucs(
    grid: List[List[int]], start: Position, goal: Position
) -> Optional[List[Position]]:
    """
    Uniform‑Cost Search on a 2‑D grid where each cell value represents
    the traversal cost (0 → free cell with cost 0, any positive int → cost).

    Returns the lowest‑cost path from start to goal (including both
    endpoints) or None if no path exists.
    """
    # ------------------------------------------------------------------
    # TODO 1 – Input validation (same as before, but also ensure all
    #          costs are non‑negative integers)
    # ------------------------------------------------------------------
    # YOUR CODE HERE ---------------------------------------------------

    # ------------------------------------------------------------------
    # TODO 2 – Implement UCS using a priority queue (heapq).
    #   * Each entry in the heap: (total_cost, position)
    #   * Keep a dict `cost_so_far` mapping Position → cheapest cost found.
    #   * Keep a dict `came_from` for path reconstruction.
    #   * When a neighbor offers a cheaper cost, push it onto the heap.
    # ------------------------------------------------------------------
    # YOUR CODE HERE ---------------------------------------------------

    # ------------------------------------------------------------------
    # TODO 3 – Reconstruct and return the cheapest path (same as BFS/DFS)
    # ------------------------------------------------------------------
    # YOUR CODE HERE ---------------------------------------------------

    # No path found
    return None
