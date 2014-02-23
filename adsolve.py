import os
import sys
sys.path.append(os.path.realpath('..')) # for running unittest
import unittest
import numbers
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from numpad.adarray import *
from numpad.adarray import _diff_recurse, _clear_tmp_product

class adsolution(adarray):
    def __init__(self, solution, residual, n_Newton=0):
        adarray.__init__(self, solution._base)
        self._residual = residual
        self._residual_ops = residual.i_ops()
        self._n_Newton = n_Newton
        self._res_norm = np.linalg.norm(residual._base)
        self._res_diff_solulion = residual.diff(solution)

    def adjoint(self, functional, s):
        assert functional.size == self.size
        J_u = self._res_diff_solulion.tocsc()

        _clear_tmp_product(self._residual, self._residual_ops)
        J_s = _diff_recurse(self._residual, s, self._residual_ops)

        c_residual = splinalg.spsolve(J_u.T, base(functional).ravel(),
                                      use_umfpack=False)
        return -(J_s.T * c_residual).reshape(s.shape)

    def tangent(self, u):
        J_u = self._res_diff_solulion.tocsr()

        _clear_tmp_product(self._residual, self._residual_ops)
        J_s = _diff_recurse(self._residual, u, self._residual_ops)

        d_self_du = -splinalg.spsolve(J_u, J_s, use_umfpack=False)
        return d_self_du.reshape(self.shape + u.shape)

    def obliviate(self):
        del self._residual
        del self._residual_ops
        del self._n_Newton
        del self._res_norm
        del self._res_diff_solulion



def solve(func, u0, args=(), kargs={},
          max_iter=10, abs_tol=1E-6, rel_tol=1E-6, verbose=True):
    u = adarray(base(u0).copy())
    for i_Newton in range(max_iter):
        res = func(u, *args, **kargs)  # TODO: how to put into adarray context?
        res_norm = np.linalg.norm(res._base, np.inf)
        if verbose:
            print('    ', i_Newton, res_norm)
        if i_Newton == 0:
            res_norm0 = res_norm
        if res_norm < max(abs_tol, rel_tol * res_norm0):
            break
        # Newton update
        J = res.diff(u).tocsr()
        minus_du = splinalg.spsolve(J, np.ravel(res._base), use_umfpack=False)
        u._base -= minus_du.reshape(u.shape)
        u = adarray(u._base)  # unlink operation history if any

    return adsolution(u, res, i_Newton + 1)



# =========================================================== #
#                                                             #
#                         unittests                           #
#                                                             #
# =========================================================== #

class _Poisson1dTest(unittest.TestCase):
    def residual(self, u, f, dx):
        res = -2 * u
        res[1:] += u[:-1]
        res[:-1] += u[1:]
        return res / dx**2 + f

    def testPoisson1d(self):
        N = 256
        dx = adarray(1. / N)

        f = ones(N-1)
        u = zeros(N-1)

        u = solve(self.residual, u, (f, dx), verbose=False)

        x = np.linspace(0, 1, N+1)[1:-1]
        self.assertAlmostEqual(0, np.abs(u._base - 0.5 * x * (1 - x)).max())

        # solve tangent equation
        dudx = u.tangent(dx)
        self.assertAlmostEqual(0, np.abs(dudx - 2 * u._base / dx._base).max())

        # solve adjoint equation
        dJdf = u.adjoint(ones(N-1), f)
        self.assertAlmostEqual(0, np.abs(dJdf - u._base).max())


class _Poisson2dTest(unittest.TestCase):
    def residual(self, u, f, dx, dy):
        res = -(2 / dx**2 + 2 / dy**2) * u
        res[1:,:] += u[:-1,:] / dx**2
        res[:-1,:] += u[1:,:] / dx**2
        res[:,1:] += u[:,:-1] / dy**2
        res[:,:-1] += u[:,1:] / dy**2
        res += f
        return res

    def testPoisson2d(self):
        #N, M = 256, 512
        N, M = 256, 64
        dx, dy = adarray([1. / N, 1. / M])

        f = ones((N-1, M-1))
        u = ones((N-1, M-1))

        u = solve(self.residual, u, (f, dx, dy), verbose=False)

        x = np.linspace(0, 1, N+1)[1:-1]
        y = np.linspace(0, 1, M+1)[1:-1]

        # solve tangent equation
        dudx = u.tangent(dx)
        dudy = u.tangent(dy)

        self.assertAlmostEqual(0,
            abs(u._base - .5 * (dudx * dx._base + dudy * dy._base)).max())

        # solve adjoint equation
        dJdf = u.adjoint(ones([N-1, M-1]), f)

        self.assertAlmostEqual(0, abs(u._base - dJdf).max())


if __name__ == '__main__':
    # a = _Poisson1dTest()
    # a.testPoisson1d()
    unittest.main()
