import unittest
import numpy as np

from .main import LagrangeInterp, R1CS_to_QAP, minimal_polynomial, validR1CSWithQAP


class TestLagrangeInterp(unittest.TestCase):
    def testLagrangeInterpSimple(self):
        poly = LagrangeInterp(np.array([0, 0, 0, 5]))

        self.assertAlmostEqual(0, poly(1))
        self.assertAlmostEqual(0, poly(2))
        self.assertAlmostEqual(0, poly(3))
        self.assertAlmostEqual(5, poly(4))

    def testLagrangeInterpHard(self):
        poly = LagrangeInterp(np.array([5, -4, 20, 5]))

        self.assertAlmostEqual(5, poly(1))
        self.assertAlmostEqual(-4, poly(2))
        self.assertAlmostEqual(20, poly(3))
        self.assertAlmostEqual(5, poly(4))


class TestR1CSToQAP(unittest.TestCase):
    def testLagrangeInterpSimple(self):
        A = np.array(
            [
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 1, 0],
                [5, 0, 0, 0, 0, 1],
            ]
        )

        actual = R1CS_to_QAP(A)

        # Taken from https://medium.com/@VitalikButerin/quadratic-arithmetic-programs-from-zero-to-hero-f6d558cea649
        expected = [
            [-5.0, 9.166, -5.0, 0.833],
            [8.0, -11.333, 5.0, -0.666],
            [0.0, 0.0, 0.0, 0.0],
            [-6.0, 9.5, -4.0, 0.5],
            [4.0, -7.0, 3.5, -0.5],
            [-1.0, 1.833, -1.0, 0.166],
        ]

        for actualRow, expectedRow in zip(actual, expected):
            for actualCell, expectedCell in zip(actualRow, expectedRow):
                self.assertAlmostEqual(actualCell, expectedCell, 2)


class TestMinimalPolynomial(unittest.TestCase):
    def testMinimalPolynomial(self):
        minPoly = minimal_polynomial(4)

        expectedMinPoly = np.polynomial.Polynomial([24, -50, 35, -10, 1])

        self.assertEqual(expectedMinPoly, minPoly)


class TestValidR1CSWithQAP(unittest.TestCase):
    def testSimple(self):
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

        S = np.array([1, 3, 35, 9, 27, 30])
        self.assertTrue(validR1CSWithQAP(A, B, C, S))

        S = np.array([2, 3, 35, 9, 27, 30])
        self.assertFalse(validR1CSWithQAP(A, B, C, S))
