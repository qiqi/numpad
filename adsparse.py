import os
import sys
import unittest
import numbers
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

sys.path.append(os.path.realpath('..')) # for running unittest

from numpad.adstate import _add_ops
from numpad.adarray import *
from numpad.adarray import __DEBUG_MODE__, _DEBUG_perturb_new
from numpad.adsolve import ResidualState, SolutionState, adsolution, solve

class sparse_matrix:
    '''
    Sparse matrix that can be automatically differentiated.
    e.g.,
    A = sparse_matrix(data, i, j, shape)
    u = spsolve(A, b)
    J = dot(c, u)
    J.diff(data)

    The constructor
    A = sparse_matrix(data, i, j, shape)
    is similar to the following in scipy.sparse
    A = csr_matrix((data, (i, j)), shape)
    '''
    def __init__(self, data, i, j, shape=None):
        self.data = data.copy()
        self.i = i.copy()
        self.j = j.copy()
        self.shape = shape
        if shape is None:
            self.shape = (i.max() + 1, j.max() + 1)

        self._base = sp.csr_matrix((self.data._base, (i, j)), shape=self.shape)

    def __mul__(self, b):
        '''
        Only implemented for a single vector b
        '''
        assert b.ndim == 1
        A_x_b = adarray(self._base * b._base)

        A_x_b.next_state(self._base, b, '*')

        data_multiplier = sp.csr_matrix((b._base[self.j],
                                         (self.i, np.arange(self.data.size))))
        A_x_b.next_state(data_multiplier, self.data, '*')

        return A_x_b


def spsolve(A, b):
    '''
    AD equivalence of scipy.sparse.linalg.spsolve.
    '''
    x = adarray(sp.linalg.spsolve(A._base, b._base))
    r = A * x - b
    return adsolution(x, r, 1)


# =========================================================== #
#                                                             #
#                         unittests                           #
#                                                             #
# =========================================================== #

if __name__ == '__main__':
    N = 100
    dx = 1. / N
    a = ones(N)
    b = ones(N-1)

    # easy way
    def resid(u):
        u = hstack([0, u, 0])
        adu = a * (u[1:] - u[:-1]) / dx
        return (adu[1:] - adu[:-1]) / dx - b

    u = solve(resid, ones(N-1))
    J = u.sum()
    adj = np.array(J.diff(a)).ravel()
    plot(adj)

    # sparse matrix way
    def tridiag(a):
        lower = a[1:-1] / dx**2
        i_lower, j_lower = np.arange(1,N-1), np.arange(N-2)

        upper = a[1:-1] / dx**2
        i_upper, j_upper = np.arange(N-2), np.arange(1,N-1)

        diag = -(a[:-1] + a[1:]) / dx**2
        i_diag, j_diag = np.arange(N-1), np.arange(N-1)

        A = sparse_matrix(hstack([lower, upper, diag]),
                          np.hstack([i_lower, i_upper, i_diag]),
                          np.hstack([j_lower, j_upper, j_diag]))
        return A

    u1 = spsolve(tridiag(a), b)
    J1 = u1.sum()
    adj1 = np.array(J1.diff(a)).ravel()
    plot(adj1)

    # finite difference
    fd = np.zeros(a.size)
    for i in range(a.size):
        a[:] = 1
        a[i] = 1 + 1E-6
        A = tridiag(a)
        du = sp.linalg.spsolve(A._base, b._base) - u._base
        fd[i] = du.sum() / 1E-6

    plot(fd)

    print('Adj - Adj1', norm(adj - adj1))
    print('Adj - fd', norm(adj - fd))
    print('Adj1 - fd', norm(adj1 - fd))
