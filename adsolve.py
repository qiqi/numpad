import os
import sys
sys.path.append(os.path.realpath('..')) # for running unittest
import unittest
import numbers
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from numpad.adarray import *
from numpad.adarray import __DEBUG_MODE__, _DEBUG_perturb_new

class adsolution(adarray):
    def __init__(self, solution, residual, n_Newton):
        adarray.__init__(self, solution._base)
        self._residual = residual
        self._residual_ops = residual.i_ops()
        self._n_Newton = n_Newton
        self._res_norm = np.linalg.norm(residual._base)
        self._res_diff_solulion = residual.diff(solution)

        _DEBUG_perturb_new(self)

    def obliviate(self):
        del self._residual
        del self._residual_ops
        del self._n_Newton
        del self._res_norm
        del self._res_diff_solulion

    # ------------------ recursive functions for differentiation --------------- #
    def _clear_tmp_product(self, i_ops):
        adarray._clear_tmp_product(self, i_ops)
        if i_ops == 0 and hasattr(self, '_residual'):
            self._residual._clear_tmp_product(self._residual_ops)
    
    def _diff_recurse(self, u, i_ops):
        if i_ops > 0 or u is self:
            return adarray._diff_recurse(self, u, i_ops)
        elif not hasattr(self, '_residual'):
            return 0

        if not hasattr(self, '_tmp_product'):
            self._tmp_product = {}
        elif 0 in self._tmp_product:  # 0 is i_ops
            return self._tmp_product[0]
    
        J_u = self._res_diff_solulion.tocsr()
        J_s = self._residual._diff_recurse(u, self._residual_ops)
        J_s = np.array(J_s.todense())

        J_u_solve = splinalg.factorized(J_u.tocsc())
        d_self_du = np.transpose([-J_u_solve(b) for b in J_s.T])
        product = np.matrix(d_self_du.reshape(J_s.shape))

        self._tmp_product[0] = product
        return product


def solve(func, u0, args=(), kargs={},
          max_iter=10, abs_tol=1E-6, rel_tol=1E-6, verbose=True):
    u = adarray(base(u0).copy())
    _DEBUG_perturb_new(u)

    for i_Newton in range(max_iter):
        res = func(u, *args, **kargs)  # TODO: how to put into adarray context?
        res_norm = np.linalg.norm(res._base, np.inf)
        if verbose:
            print('    ', i_Newton, res_norm)
        if i_Newton == 0:
            res_norm0 = res_norm
        if res_norm < max(abs_tol, rel_tol * res_norm0):
            return adsolution(u, res, i_Newton + 1)
        # Newton update
        J = res.diff(u).tocsr()
        minus_du = splinalg.spsolve(J, np.ravel(res._base), use_umfpack=False)
        u._base -= minus_du.reshape(u.shape)
        u = adarray(u._base)  # unlink operation history if any
        _DEBUG_perturb_new(u)
    # not converged
    return adsolution(u, res, np.inf)



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
        dudx = np.array(u.diff(dx)).reshape(u.shape)
        self.assertAlmostEqual(0, np.abs(dudx - 2 * u._base / dx._base).max())

        # solve adjoint equation
        J = u.sum()
        dJdf = J.diff(f)
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
        dudx = np.array(u.diff(dx)).reshape(u.shape)
        dudy = np.array(u.diff(dy)).reshape(u.shape)

        self.assertAlmostEqual(0,
            abs(2 * u._base - (dudx * dx._base + dudy * dy._base)).max())

        # # solve adjoint equation
        # J = u.sum()
        # dJdf = u.diff(f)

        # self.assertAlmostEqual(0, abs(u._base - dJdf).max())


if __name__ == '__main__':
    # a = _Poisson1dTest()
    # a.testPoisson1d()
    unittest.main()
