# ------------------------------------------------------------
# test_bfs.py
# ------------------------------------------------------------
# Run this file from the command line (or via the IDE) to
# automatically check the student's bfs implementation.
# ------------------------------------------------------------

import sys
import importlib.util
from pathlib import Path
from typing import List, Tuple


# ----------------------------------------------------------------
# Helper to import a module from an arbitrary path (used for both
# the template version (student) and the reference solution).
# ----------------------------------------------------------------
def import_module_from_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


# ----------------------------------------------------------------
# Configuration – where the files live (adjust if you store them
# elsewhere)
# ----------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
STUDENT_FILE = SCRIPT_DIR / "bfs_template.py"

TEST_MODULE_NAME = "bfs_impl"

# ----------------------------------------------------------------
# Load the student's implementation (they must have defined `bfs`)
# ----------------------------------------------------------------
student_mod = import_module_from_path(TEST_MODULE_NAME, str(STUDENT_FILE))
if not hasattr(student_mod, "bfs"):
    sys.exit("ERROR: bfs function not found in the student's file.")

bfs_student = getattr(student_mod, "bfs")

# ----------------------------------------------------------------
# Test cases
# ----------------------------------------------------------------
TestGrid = List[List[int]]
Position = Tuple[int, int]


def run_test(grid: TestGrid, start: Position, goal: Position, expected) -> bool:
    """
    Executes bfs on the student's code and compares the result with
    the expected path (order matters).  Returns True if the test passes.
    """
    try:
        result = bfs_student(grid, start, goal)
    except Exception as e:
        print(f"❌ Exception raised during bfs call: {e}")
        return False

    if result != expected:
        print("❌ Test failed:")
        print(f"   Grid   : {grid}")
        print(f"   Start  : {start}")
        print(f"   Goal   : {goal}")
        print(f"   Expected: {expected}")
        print(f"   Got     : {result}")
        return False
    return True


def main():
    all_passed = True

    # ------------------------------------------------------------
    # 1️⃣ Simple 3×3 open grid – shortest path is the Manhattan line
    # ------------------------------------------------------------
    grid1 = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    start1 = (0, 0)
    goal1 = (2, 2)
    expected1 = [
        (0, 0),
        (1, 0),
        (2, 0),
        (2, 1),
        (2, 2),
    ]  # one of many possible shortest paths
    # Because BFS explores neighbours in Up‑Down‑Left‑Right order we know the exact path.
    all_passed &= run_test(grid1, start1, goal1, expected1)

    # ------------------------------------------------------------
    # 2️⃣ Grid with obstacles – path must go around the wall
    # ------------------------------------------------------------
    grid2 = [
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
    ]
    start2 = (0, 0)
    goal2 = (3, 3)
    expected2 = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (3, 2), (3, 3)]
    all_passed &= run_test(grid2, start2, goal2, expected2)

    # ------------------------------------------------------------
    # 3️⃣ No possible path (goal isolated by walls)
    # ------------------------------------------------------------
    grid3 = [
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0],
    ]
    start3 = (0, 0)
    goal3 = (2, 2)
    expected3 = None
    all_passed &= run_test(grid3, start3, goal3, expected3)

    # ------------------------------------------------------------
    # 4️⃣ Invalid input – start outside grid → should raise ValueError
    # ------------------------------------------------------------
    try:
        bfs_student([[0]], (5, 0), (0, 0))
        print("❌ Expected ValueError for start out of bounds, but none raised.")
        all_passed = False
    except ValueError:
        pass  # correct behavior
    except Exception as e:
        print(f"❌ Unexpected exception type for out‑of‑bounds start: {e}")
        all_passed = False

    # ------------------------------------------------------------
    # 5️⃣ Invalid input – start on a wall → should raise ValueError
    # ------------------------------------------------------------
    try:
        bfs_student([[1]], (0, 0), (0, 0))
        print("❌ Expected ValueError for start on a wall, but none raised.")
        all_passed = False
    except ValueError:
        pass
    except Exception as e:
        print(f"❌ Unexpected exception for start on wall: {e}")
        all_passed = False

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    if all_passed:
        print("\n✅ All tests passed!  Your BFS implementation works correctly.")
    else:
        print(
            "\n⚠️ Some tests failed.  Review the error messages above, fix your code, and run the tests again."
        )


if __name__ == "__main__":
    main()
