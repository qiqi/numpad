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

class csr_matrix:
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
    def __init__(self, data, shape=None):
        if len(data) == 3:
            data, col_ind, row_ptr = data
            self._value = sp.csr_matrix((data._value, col_ind, row_ptr),
                                       shape=shape)
            self.i, self.j = self._value.nonzero()
            self.data = data.copy()
        elif len(data) == 2:
            data, ij = data
            i = np.asarray(value(ij[0]), int)
            j = np.asarray(value(ij[1]), int)

            self.data = data.copy()
            self._value = sp.csr_matrix((data._value, (i, j)), shape=shape)
            self.i, self.j = i.copy(), j.copy()
        else:
            raise NotImplementedError()

        self.shape = shape
        if shape is None:
            self.shape = (self.i.max() + 1, self.j.max() + 1)

    def __mul__(self, b):
        '''
        Only implemented for a single vector b
        '''
        if b.ndim == 1:
            A_x_b = adarray(self._value * b._value)
            A_x_b.next_state(self._value, b, '*')

            data_multiplier = sp.csr_matrix((b._value[self.j],
                                  (self.i, np.arange(self.data.size))))
            A_x_b.next_state(data_multiplier, self.data, '*')
            return A_x_b
        else:
            shape = b.shape[1:]
            b = b.reshape([b.shape[0], -1])
            a = transpose([self * bi for bi in b.T])
            return a.reshape((a.shape[0],) + shape)



def spsolve(A, b):
    '''
    AD equivalence of scipy.sparse.linalg.spsolve.
    '''
    x = adarray(sp.linalg.spsolve(A._value.tocsr(), b._value))
    r = A * x - b
    return adsolution(x, r, 1)


# =========================================================== #
#                                                             #
#                         unittests                           #
#                                                             #
# =========================================================== #

if __name__ == '__main__':
    import pylab
    # data = ones([3, 2, 2])
    # data[:,0,1] = 0
    # row_ptr = np.array([0,1,2,3], int)
    # col_ind = np.array([0,1,2], int)
    # A = bsr_matrix((data, col_ind, row_ptr))

    # b = ones(6)
    # x = spsolve(A, b)

    # x.diff(data)

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
    pylab.plot(adj)

    # sparse matrix way
    def tridiag(a):
        lower = a[1:-1] / dx**2
        i_lower, j_lower = np.arange(1,N-1), np.arange(N-2)

        upper = a[1:-1] / dx**2
        i_upper, j_upper = np.arange(N-2), np.arange(1,N-1)

        diag = -(a[:-1] + a[1:]) / dx**2
        i_diag, j_diag = np.arange(N-1), np.arange(N-1)

        A = csr_matrix([hstack([lower, upper, diag]),
                        (np.hstack([i_lower, i_upper, i_diag]),
                         np.hstack([j_lower, j_upper, j_diag]))])
        return A

    u1 = spsolve(tridiag(a), b)
    J1 = u1.sum()
    adj1 = np.array(J1.diff(a)).ravel()
    pylab.plot(adj1)

    # finite difference
    fd = np.zeros(a.size)
    for i in range(a.size):
        a[:] = 1
        a[i] = 1 + 1E-6
        A = tridiag(a)
        du = sp.linalg.spsolve(A._value, b._value) - u._value
        fd[i] = du.sum() / 1E-6

    pylab.plot(fd)

    print('Adj - Adj1', np.linalg.norm(adj - adj1))
    print('Adj - fd', np.linalg.norm(adj - fd))
    print('Adj1 - fd', np.linalg.norm(adj1 - fd))

    # second test
    u1 = tridiag(a) * transpose([[b, b], [b,b]])
