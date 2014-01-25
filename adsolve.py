import unittest
import numbers
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from adarray import *
from adarray import _base

class adsolution(adarray):
    def __init__(self, solution, residual, n_Newton=0):
        adarray.__init__(self, solution._base)
        self._residual = residual
        self._residual_ops = residual.i_ops()
        self._n_Newton = n_Newton
        self._res_norm = np.linalg.norm(residual._base)
        self._res_diff_solulion = residual.diff(solution)

    def adjoint(self, functional, u):
        assert functional.size == self.size
        J_u = self._res_diff_solulion.tocsc()
        J_s = self._residual._diff_recurse(u, self._residual_ops)
        c_residual = splinalg.spsolve(J_u.T, _base(functional).ravel(),
                                      use_umfpack=False)
        return -(J_s.T * c_residual).reshape(u.shape)

    def tangent(self, u):
        J_u = self._res_diff_solulion.tocsr()
        J_s = self._residual._diff_recurse(u, self._residual_ops)
        d_self_du = -splinalg.spsolve(J_u, J_s, use_umfpack=False)
        return d_self_du.reshape(self.shape + u.shape)


def solve(func, u0, args=(), kargs={},
          max_iter=8, abs_tol=1E-6, rel_tol=1E-6, verbose=True):
    u = adarray(_base(u0.copy()))
    for i_Newton in range(max_iter):
        res = func(u, *args, **kargs)  # TODO: how to put adarray context?
        res_norm = np.linalg.norm(res._base)
        if verbose:
            print('    ', i_Newton, res_norm)
        if i_Newton == 0:
            res_norm0 = res_norm
        if res_norm < max(abs_tol, rel_tol * res_norm0):
            break
        # Newton update
        J = res.diff(u).tocsr()
        minus_du = splinalg.spsolve(J, ravel(res._base), use_umfpack=False)
        u._base -= minus_du.reshape(u.shape)
        u = adarray(u._base)  # unlink operation history if any

    return adsolution(u, res, i_Newton + 1)



# =========================================================== #
#                                                             #
#                         unittests                           #
#                                                             #
# =========================================================== #

class Poisson1dTest(unittest.TestCase):
    def residual(self, u, f, dx):
        res = -2 * u
        res[1:] += u[:-1]
        res[:-1] += u[1:]
        return res / dx**2 + f

    def testPoisson1d(self):
        N = 496
        dx = adarray(1 / N)

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


class Poisson2dTest(unittest.TestCase):
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
        dx, dy = adarray([1 / N, 1 / M])

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
    unittest.main()
