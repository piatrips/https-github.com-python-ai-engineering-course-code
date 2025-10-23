from __future__ import annotations
import random
import time
from typing import Iterable

def compare_membership(n: int = 5000, checks: int = 2000) -> dict:
    """Compare membership test performance for a list vs a dict.

    Returns a dict with measured times in seconds.
    """
    data_list = list(range(n))
    data_dict = {i: True for i in range(n)}

    # prepare random lookups (some present, some not)
    lookups = [random.randint(0, int(n * 1.5)) for _ in range(checks)]

    # measure list membership
    t0 = time.perf_counter()
    for x in lookups:
        _ = x in data_list
    t_list = time.perf_counter() - t0

    # measure dict membership
    t0 = time.perf_counter()
    for x in lookups:
        _ = x in data_dict
    t_dict = time.perf_counter() - t0

    return {
        "n": n,
        "checks": checks,
        "list_time": t_list,
        "dict_time": t_dict,
        "dict_faster": t_dict < t_list,
    }

if __name__ == "__main__":
    res = compare_membership()
    print("Membership comparison:")
    for k, v in res.items():
        print(f"  {k}: {v}")
