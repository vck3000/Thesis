"""
Implementing algorithm described in: https://medium.com/@VitalikButerin/quadratic-arithmetic-programs-from-zero-to-hero-f6d558cea649

For the circuit:

def qeval(x):
    y = x**3
    return x + y + 5

We flatten it to:

sym_1 = x * x
y = sym_1 * x
sym_2 = y + x
~out = sym_2 + 5

Then we convert to R1CS (Rank-1 Constraint System)

A
[0, 1, 0, 0, 0, 0]
[0, 0, 0, 1, 0, 0]
[0, 1, 0, 0, 1, 0]
[5, 0, 0, 0, 0, 1]
B
[0, 1, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 0]
[1, 0, 0, 0, 0, 0]
[1, 0, 0, 0, 0, 0]
C
[0, 0, 0, 1, 0, 0]
[0, 0, 0, 0, 1, 0]
[0, 0, 0, 0, 0, 1]
[0, 0, 1, 0, 0, 0]

S (witness)
[1, 3, 35, 9, 27, 30]

Such that for each row in A, B, C, the following holds:
dot(A, S) * dot(B, S) - dot(C, S) = 0
"""

import numpy as np
from numpy.polynomial import Polynomial

from functools import reduce
from typing import List

# Setup
def main():
    A = np.array(
        [
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [5, 0, 0, 0, 0, 1],
        ]
    )

    B = np.array(
        [
            [0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
        ]
    )

    C = np.array(
        [
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
        ]
    )

    assert A.shape == B.shape == C.shape

    # S = ['~one', 'x', '~out', 'sym_1', 'y', 'sym_2']
    S = np.array([1, 3, 35, 9, 27, 30])

    assert S.size == A.shape[1]

    # Verify S is a correct witness manually
    # for i in range(rows):
    #     assert np.dot(A[i], S) * np.dot(B[i], S) - np.dot(C[i], S) == 0

    res = validR1CSWithQAP(A, B, C, S, verbose=True)
    if res:
        print("Valid R1CS!")
    else:
        print("Failed R1CS QAP check.")


def validR1CSWithQAP(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, S: np.ndarray, verbose=False
):
    # Get polynomials
    A_poly = R1CS_to_QAP(A)
    B_poly = R1CS_to_QAP(B)
    C_poly = R1CS_to_QAP(C)

    (rows, _) = A.shape

    # Now to verify the QAP
    # A(x).S * B(x).S - C(x).S = t(x) = H * Z(x)
    # where Z(x) is the minimal polynomial with roots of 0 at each gate.

    t = np.dot(A_poly, S) * np.dot(B_poly, S) - np.dot(C_poly, S)

    Z = minimal_polynomial(rows)

    if verbose:
        print("Quotient:")
        print(t // Z)
        print()
        print("Remainder:")
        print(t % Z)
        print()

    for c in (t % Z).coef:
        if abs(c) > 1e-7:
            return False

    return True


# Convert R1CS to QAP via Lagrange interpolation
def R1CS_to_QAP(arr) -> List[np.polynomial.Polynomial]:
    polynomials = []

    rows, cols = arr.shape

    for i in range(cols):
        polynomials.append(LagrangeInterp(arr[:, i]))

    return polynomials


def LagrangeInterp(arr):
    """
    Calculates the Lagrangian Interpolation for an array.

    E.g. [0, 0, 0, 5] -> a polynomial that x({1, 2, 3}) == 0 and x(4) == 5.

    For each point, a polynomial for that point is generated, with all other points set to 0.
    Then, linearly add these polynomials.
    """
    # np.ones here so addition with the identity works nicely.
    poly = Polynomial(np.zeros(arr.size))

    for x in range(arr.size):
        roots = list(range(1, arr.size + 1))
        roots.pop(x)

        new_poly = create_polynomial_with_roots(roots)

        # Scale the polynomial so that f(x) == arr[x]
        # x = A.x where A is the amplitude scaling factor
        current_y = new_poly(x + 1)
        A = arr[x] / current_y
        new_poly = new_poly * A

        poly += new_poly

    return poly


def create_polynomial_with_roots(roots):
    # Turn set of root values into a list of polynomial fragments such as (x-2), (x-3), ...
    polynomial_fragments = list(map(lambda root: Polynomial([-root, 1]), roots))

    # Multiply all the polynomial fragments, i.e. (x-2) * (x-3) * ...
    poly = reduce(lambda poly1, poly2: poly1 * poly2, polynomial_fragments)

    return poly


def minimal_polynomial(degree):
    roots = list(range(1, degree + 1))

    return create_polynomial_with_roots(roots)


if __name__ == "__main__":
    main()
