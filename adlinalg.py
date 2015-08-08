import os
import sys
import unittest
import numbers
import numpy as np

sys.path.append(os.path.realpath('..')) # for running unittest

from numpad.adarray import *
from numpad.adarray import __DEBUG_MODE__, _DEBUG_perturb_new
from numpad.adsolve import adsolution
import numpad.adrandom as random

def solve(A, b):
    '''
    AD equivalence of linalg.solve
    '''
    assert A.ndim == 2 and b.shape[0] == A.shape[0]
    x = adarray(np.linalg.solve(value(A), value(b)))
    r = dot(A, x) - b
    return adsolution(x, r, 1)

# =========================================================== #
#                                                             #
#                         unittests                           #
#                                                             #
# =========================================================== #
class _AnalyticalInverseTest(unittest.TestCase):
    def testDiagonalPerturbation(self):
        import pylab
        N = 2

        A_additional_diag = array(1)
        A = random.random([N, N]) + A_additional_diag * eye(N)

        b = eye(N)
        Ainv = solve(A, b)

        Ainv_diff_A_diag = Ainv.diff(A_additional_diag)
        Ainv_diff_A_diag = np.array(Ainv_diff_A_diag.todense()).reshape([N, N])

        Ainv_diff_A_diag_analytical = -np.dot(value(Ainv), value(Ainv))
        difference = Ainv_diff_A_diag - Ainv_diff_A_diag_analytical
        self.assertAlmostEqual(abs(difference).max(), 0)

if __name__ == '__main__':
        unittest.main()
