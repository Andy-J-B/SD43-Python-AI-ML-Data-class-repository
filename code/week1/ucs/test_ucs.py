# ------------------------------------------------------------
# test_ucs.py
# ------------------------------------------------------------
# Tests for the Uniform‑Cost Search implementation.
# ------------------------------------------------------------

import sys
import importlib.util
from pathlib import Path
from typing import List, Tuple, Optional


# ----------------------------------------------------------------
# Helper to import a module from file
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
STUDENT_FILE = SCRIPT_DIR / "ucs_template.py"

MODULE_NAME = "ucs_impl"

student_mod = import_module(MODULE_NAME, str(STUDENT_FILE))
if not hasattr(student_mod, "ucs"):
    sys.exit("ERROR: ucs function not found in the student's file.")
ucs_student = getattr(student_mod, "ucs")


# ----------------------------------------------------------------
# Test helper
# ----------------------------------------------------------------
Grid = List[List[int]]
Pos = Tuple[int, int]


def run_test(grid: Grid, start: Pos, goal: Pos, expected: Optional[List[Pos]]) -> bool:
    try:
        result = ucs_student(grid, start, goal)
    except Exception as e:
        print(f"❌ Exception while calling ucs: {e}")
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
    # 1️⃣ Uniform costs (all 1) – path should be same as BFS shortest path
    # ------------------------------------------------------------
    grid1 = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    start1 = (0, 0)
    goal1 = (2, 2)
    expected1 = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]  # Manhattan shortest path
    all_ok &= run_test(grid1, start1, goal1, expected1)

    # ------------------------------------------------------------
    # 2️⃣ Different terrain costs – the algorithm must go around the
    #     expensive cell (cost 9) even if it makes the path longer.
    # ------------------------------------------------------------
    grid2 = [
        [1, 1, 1, 1],
        [1, 9, 9, 1],
        [1, 1, 1, 1],
    ]
    start2 = (0, 0)
    goal2 = (2, 3)
    # Cheapest route goes down the left side then across the bottom:
    expected2 = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3)]
    all_ok &= run_test(grid2, start2, goal2, expected2)

    # ------------------------------------------------------------
    # 3️⃣ No possible path (walls have cost = -1? not allowed – we use very high cost)
    #     Here we simply set cost = 0 for walls and treat them as impassable
    #     by surrounding with a huge cost (e.g., 9999).  The algorithm should
    #     return None.
    # ------------------------------------------------------------
    grid3 = [
        [1, 9999, 1],
        [9999, 9999, 9999],
        [1, 9999, 1],
    ]
    start3 = (0, 0)
    goal3 = (2, 2)
    expected3 = None
    all_ok &= run_test(grid3, start3, goal3, expected3)

    # ------------------------------------------------------------
    # 4️⃣ Invalid input – negative cost cell (should raise ValueError)
    # ------------------------------------------------------------
    try:
        ucs_student([[1, -5], [1, 1]], (0, 0), (1, 1))
        print("❌ Expected ValueError for negative cost, but none raised.")
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
        print("\n✅ All UCS tests passed!")
    else:
        print("\n⚠️ Some UCS tests failed – see messages above.")


if __name__ == "__main__":
    main()
