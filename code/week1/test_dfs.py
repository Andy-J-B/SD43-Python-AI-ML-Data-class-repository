# ------------------------------------------------------------
# test_dfs.py
# ------------------------------------------------------------
# Runs a battery of tests on the student's DFS implementation.
# ------------------------------------------------------------

import sys
import importlib.util
from pathlib import Path
from typing import List, Tuple, Optional


# ----------------------------------------------------------------
# Helper to import a module from a given file path
# ----------------------------------------------------------------
def import_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


# ----------------------------------------------------------------
# Paths (adjust if you move the files)
# ----------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
STUDENT_FILE = SCRIPT_DIR / "dfs_template.py"

MODULE_NAME = "dfs_impl"

student_mod = import_module(MODULE_NAME, str(STUDENT_FILE))
if not hasattr(student_mod, "dfs"):
    sys.exit("ERROR: dfs function not found in the student's file.")
dfs_student = getattr(student_mod, "dfs")


# ----------------------------------------------------------------
# Test utilities
# ----------------------------------------------------------------
Grid = List[List[int]]
Pos = Tuple[int, int]


def run_test(grid: Grid, start: Pos, goal: Pos, expected: Optional[List[Pos]]) -> bool:
    try:
        result = dfs_student(grid, start, goal)
    except Exception as e:
        print(f"❌ Exception while calling dfs: {e}")
        return False

    if result != expected:
        print("❌ Test failed")
        print(f"   Grid    : {grid}")
        print(f"   Start   : {start}")
        print(f"   Goal    : {goal}")
        print(f"   Expected: {expected}")
        print(f"   Got     : {result}")
        return False
    return True


def main():
    all_ok = True

    # ------------------------------------------------------------
    # 1️⃣ Simple open 3×3 grid – any depth‑first path is fine.
    # ------------------------------------------------------------
    grid1 = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    start1 = (0, 0)
    goal1 = (2, 2)
    # With the neighbour order Up‑Down‑Left‑Right the first DFS path is:
    expected1 = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    all_ok &= run_test(grid1, start1, goal1, expected1)

    # ------------------------------------------------------------
    # 2️⃣ Grid with a wall – path must go around it.
    # ------------------------------------------------------------
    grid2 = [
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]
    start2 = (0, 0)
    goal2 = (0, 2)
    expected2 = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)]
    all_ok &= run_test(grid2, start2, goal2, expected2)

    # ------------------------------------------------------------
    # 3️⃣ No path possible
    # ------------------------------------------------------------
    grid3 = [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ]
    start3 = (0, 0)
    goal3 = (2, 2)
    expected3 = None
    all_ok &= run_test(grid3, start3, goal3, expected3)

    # ------------------------------------------------------------
    # 4️⃣ Invalid inputs – must raise ValueError
    # ------------------------------------------------------------
    try:
        dfs_student([[0]], (5, 0), (0, 0))
        print("❌ Expected ValueError for start OOB, but none raised.")
        all_ok = False
    except ValueError:
        pass
    except Exception as e:
        print(f"❌ Unexpected exception type: {e}")
        all_ok = False

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    if all_ok:
        print("\n✅ All DFS tests passed!")
    else:
        print("\n⚠️ Some DFS tests failed – see messages above.")


if __name__ == "__main__":
    main()
