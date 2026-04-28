# ------------------------------------------------------------
# dfs_solution.py
# ------------------------------------------------------------
# Reference solution for the DFS project.
# ------------------------------------------------------------

from typing import List, Tuple, Optional, Set

Position = Tuple[int, int]


def dfs(
    grid: List[List[int]], start: Position, goal: Position
) -> Optional[List[Position]]:
    """
    Depth‑First Search on a 2‑D grid (4‑directional moves).
    Returns the first path found from start to goal,
    or None if no path exists.
    """
    # ---------- Input validation ----------
    if not grid or not grid[0]:
        raise ValueError("Grid must be a non‑empty 2‑D list.")
    rows, cols = len(grid), len(grid[0])

    # Ensure rectangular grid
    for r in grid:
        if len(r) != cols:
            raise ValueError("All rows must have the same length.")

    def inside(p: Position) -> bool:
        r, c = p
        return 0 <= r < rows and 0 <= c < cols

    if not inside(start):
        raise ValueError("Start position out of bounds.")
    if not inside(goal):
        raise ValueError("Goal position out of bounds.")
    if grid[start[0]][start[1]] != 0:
        raise ValueError("Start position is an obstacle.")
    if grid[goal[0]][goal[1]] != 0:
        raise ValueError("Goal position is an obstacle.")

    # ---------- DFS ----------
    stack: List[Position] = [start]
    came_from: dict[Position, Optional[Position]] = {start: None}
    visited: Set[Position] = {start}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

    while stack:
        current = stack.pop()

        if current == goal:
            break

        for dr, dc in directions:
            neighbor = (current[0] + dr, current[1] + dc)
            if inside(neighbor) and grid[neighbor[0]][neighbor[1]] == 0:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
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
