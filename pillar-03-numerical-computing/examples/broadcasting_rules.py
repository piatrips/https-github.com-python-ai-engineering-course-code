"""
Demonstrate NumPy broadcasting rules with examples and small explanations.
"""
from __future__ import annotations
import numpy as np

def examples():
    # 1. scalar and array
    a = np.array([1, 2, 3])
    s = 10
    print("scalar + array:", s + a)  # scalar broadcast to shape (3,)

    # 2. array and column vector
    x = np.arange(6).reshape(2, 3)  # shape (2,3)
    v = np.array([10, 20, 30])
    print("row-wise add:", x + v)  # v broadcast along rows

    # 3. column vector broadcasting
    col = np.array([[1], [2]])  # shape (2,1)
    print("column broadcast:", col + v)  # col broadcast along columns -> shape (2,3)

    # 4. explicit broadcasting with newaxis
    a = np.array([0, 1, 2])
    b = np.array([0, 10])
    # Want outer-sum -> use broadcasting by reshaping
    print("outer sum:", a[:, np.newaxis] + b)  # result shape (3,2)

    # 5. incompatible shapes raise an error
    try:
        bad = np.ones((3,2)) + np.ones((4,))
    except ValueError as e:
        print("incompatible shapes error:", e)


if __name__ == '__main__':
    examples()