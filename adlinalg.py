import os
import sys
import unittest
import numbers
import pylab
import numpy as np

sys.path.append(os.path.realpath('..')) # for running unittest

from numpad.adarray import *
from numpad.adarray import __DEBUG_MODE__, _DEBUG_perturb_new
from numpad.adsolve import adsolution

def solve(A, b):
    '''
    AD equivalence of linalg.solve
    '''
    assert A.ndim == 2 and b.shape[0] == A.shape[0]
    x = adarray(np.linalg.solve(A, b))
    r = dot(A, x) - b
    return adsolution(x, r, 1)

# =========================================================== #
#                                                             #
#                         unittests                           #
#                                                             #
# =========================================================== #

class _AnalyticalInverseTest(unittest.TestCase):
    def testInverseDiagPert(self):
        N = 10

        A_additional_diag = 1
        A = random([N, N]) + A_additional_diag * eye(N)

        b = eye(N)
        Ainv = solve(A, b)

        Ainv_diff_A_diag = Ainv.diff(A_additional_diag).to_dense()
        Ainv_diff_A_diag = np.array(Ainv_diff_A_diag).reshape([N, N])

        Ainv_diff_A_diag_analytical = dot(value(Ainv), value(Ainv))

if __name__ == '__main__':
    unittest.main()
