# ------------------------------------------------------------
# bfs_solution.py
# ------------------------------------------------------------
# This is the *reference* implementation that the class will
# arrive at after completing the TODOs in bfs_template.py.
# ------------------------------------------------------------

from collections import deque
from typing import List, Tuple, Optional

Position = Tuple[int, int]


def bfs(
    grid: List[List[int]], start: Position, goal: Position
) -> Optional[List[Position]]:
    """
    Breadth‑First Search on a 2‑D grid.  Returns the shortest
    path from start to goal (including both endpoints) or None
    if no path exists.
    """
    # ---------- Input validation ----------
    if not grid or not grid[0]:
        raise ValueError("Grid must be a non‑empty 2‑D list.")
    rows, cols = len(grid), len(grid[0])

    # Ensure grid is rectangular
    for r in grid:
        if len(r) != cols:
            raise ValueError("All rows in the grid must have the same length.")

    def inside(p: Position) -> bool:
        r, c = p
        return 0 <= r < rows and 0 <= c < cols

    if not inside(start):
        raise ValueError("Start position out of grid bounds.")
    if not inside(goal):
        raise ValueError("Goal position out of grid bounds.")
    if grid[start[0]][start[1]] != 0:
        raise ValueError("Start position is an obstacle.")
    if grid[goal[0]][goal[1]] != 0:
        raise ValueError("Goal position is an obstacle.")

    # ---------- BFS ----------
    frontier = deque([start])
    came_from: dict[Position, Optional[Position]] = {start: None}
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while frontier:
        current = frontier.popleft()

        if current == goal:  # Goal found – stop searching
            break

        for dr, dc in directions:
            neighbor = (current[0] + dr, current[1] + dc)
            if inside(neighbor) and grid[neighbor[0]][neighbor[1]] == 0:
                if neighbor not in came_from:  # not visited yet
                    frontier.append(neighbor)
                    came_from[neighbor] = current

    # ---------- Reconstruct path ----------
    if goal not in came_from:
        # Goal never reached
        return None

    path: List[Position] = []
    cur: Position = goal
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]  # type: ignore  # mypy safe because of the check above
    path.reverse()  # now start → goal
    return path
