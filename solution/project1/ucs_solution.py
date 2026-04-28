# ------------------------------------------------------------
# ucs_solution.py
# ------------------------------------------------------------
# Reference solution for Uniform‑Cost Search.
# ------------------------------------------------------------

import heapq
from typing import List, Tuple, Optional, Dict

Position = Tuple[int, int]


def ucs(
    grid: List[List[int]], start: Position, goal: Position
) -> Optional[List[Position]]:
    """
    Uniform‑Cost Search on a 2‑D grid.
    Each cell value must be a non‑negative integer representing the cost
    to *enter* that cell.
    Returns the minimum‑cost path from start to goal, or None if unreachable.
    """
    # ---------- Input validation ----------
    if not grid or not grid[0]:
        raise ValueError("Grid must be a non‑empty 2‑D list.")
    rows, cols = len(grid), len(grid[0])

    # Ensure rectangular grid and non‑negative costs
    for r in grid:
        if len(r) != cols:
            raise ValueError("All rows must have the same length.")
        for val in r:
            if not isinstance(val, int) or val < 0:
                raise ValueError("Grid costs must be non‑negative integers.")

    def inside(p: Position) -> bool:
        r, c = p
        return 0 <= r < rows and 0 <= c < cols

    if not inside(start):
        raise ValueError("Start out of bounds.")
    if not inside(goal):
        raise ValueError("Goal out of bounds.")

    # Starting cell cost is counted **when we step into it**, so the
    # initial accumulated cost is the cost of the start cell itself.
    if grid[start[0]][start[1]] != 0:
        # It is legal for the start cell to have a cost >0; we just include it.
        pass

    # ---------- UCS ----------
    frontier: List[Tuple[int, Position]] = []
    heapq.heappush(frontier, (grid[start[0]][start[1]], start))

    came_from: Dict[Position, Optional[Position]] = {start: None}
    cost_so_far: Dict[Position, int] = {start: grid[start[0]][start[1]]}

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while frontier:
        current_cost, current = heapq.heappop(frontier)

        if current == goal:
            break

        for dr, dc in directions:
            nr, nc = current[0] + dr, current[1] + dc
            neighbor = (nr, nc)
            if not inside(neighbor):
                continue
            # Cost to move into neighbor = current accumulated + neighbor's cell cost
            new_cost = current_cost + grid[nr][nc]
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                heapq.heappush(frontier, (new_cost, neighbor))
                came_from[neighbor] = current

    # ---------- Reconstruct path ----------
    if goal not in came_from:
        return None

    path: List[Position] = []
    cur: Position = goal
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]  # type: ignore
    path.reverse()
    return path
