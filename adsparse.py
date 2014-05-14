import os
import sys
import unittest
import numbers
import pylab
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
            self._base = sp.csr_matrix((data._base, col_ind, row_ptr),
                                       shape=shape)
            self.i, self.j = self._base.nonzero()
            self.data = data.copy()
        elif len(data) == 2:
            data, ij = data
            i = np.asarray(base(ij[0]), int)
            j = np.asarray(base(ij[1]), int)

            self.data = data.copy()
            self._base = sp.csr_matrix((data._base, (i, j)), shape=shape)
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
        assert b.ndim == 1
        A_x_b = adarray(self._base * b._base)

        A_x_b.next_state(self._base, b, '*')

        data_multiplier = sp.csr_matrix((b._base[self.j],
                                         (self.i, np.arange(self.data.size))))
        A_x_b.next_state(data_multiplier, self.data, '*')

        return A_x_b


class bsr_matrix:
    '''
    Sparse matrix that can be automatically differentiated.
    e.g.,
    A = bsr_matrix(data, i, j, shape)
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
            # construct dummy matrix just to figure out row and colum indices
            # TODO: make it more efficient
            dummy = sp.bsr_matrix((np.ones(data.shape), col_ind, row_ptr),
                                   shape=shape)
            self.i, self.j = dummy.nonzero()

            self.data = data.copy()
            self._base = sp.bsr_matrix((data._base, col_ind, row_ptr),
                                       shape=shape)
        elif len(data) == 2:
            data, ij = data
            i, j = ij

            self.data = data.copy()
            self._base = sp.bsr_matrix((data._base, (i, j)), shape=shape)
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
        assert b.ndim == 1
        A_x_b = adarray(self._base * b._base)

        A_x_b.next_state(self._base, b, '*')

        data_multiplier = sp.csr_matrix((b._base[self.j],
                                         (self.i, np.arange(self.data.size))))
        A_x_b.next_state(data_multiplier, self.data, '*')

        return A_x_b

    def __sub__(self, B):
        # HACK: only support the case when A and B have
        # the same sparsity pattern
        A = self
        assert A.shape == B.shape
        assert (A._base.indices == B._base.indices).all()
        assert (A._base.indptr == B._base.indptr).all()
        return bsr_matrix((A.data - B.data, A._base.indices, A._base.indptr),
                          A.shape)

    def __mul__(self, B):
        A = self
        dummy_A = sp.csr_matrix((np.ones(A.i.shape), (A.i, A.j)))
        dummy_B = sp.csr_matrix((np.ones(B.i.shape), (B.i, B.j)))
        dummy_AB = dummy_A * dummy_B
        i, j = dummy_AB.nonzero()



def spsolve(A, b):
    '''
    AD equivalence of scipy.sparse.linalg.spsolve.
    '''
    x = adarray(sp.linalg.spsolve(A._base.tocsr(), b._base))
    r = A * x - b
    return adsolution(x, r, 1)


# =========================================================== #
#                                                             #
#                         unittests                           #
#                                                             #
# =========================================================== #

if __name__ == '__main__':
    data = ones([3, 2, 2])
    data[:,0,1] = 0
    row_ptr = array([0,1,2,3], int)
    col_ind = array([0,1,2], int)
    A = bsr_matrix((data, col_ind, row_ptr))

    b = ones(6)
    x = spsolve(A, b)

    # x.diff(data)

    # N = 100
    # dx = 1. / N
    # a = ones(N)
    # b = ones(N-1)

    # # easy way
    # def resid(u):
    #     u = hstack([0, u, 0])
    #     adu = a * (u[1:] - u[:-1]) / dx
    #     return (adu[1:] - adu[:-1]) / dx - b

    # u = solve(resid, ones(N-1))
    # J = u.sum()
    # adj = np.array(J.diff(a)).ravel()
    # pylab.plot(adj)

    # # sparse matrix way
    # def tridiag(a):
    #     lower = a[1:-1] / dx**2
    #     i_lower, j_lower = np.arange(1,N-1), np.arange(N-2)

    #     upper = a[1:-1] / dx**2
    #     i_upper, j_upper = np.arange(N-2), np.arange(1,N-1)

    #     diag = -(a[:-1] + a[1:]) / dx**2
    #     i_diag, j_diag = np.arange(N-1), np.arange(N-1)

    #     A = csr_matrix(hstack([lower, upper, diag]),
    #                       np.hstack([i_lower, i_upper, i_diag]),
    #                       np.hstack([j_lower, j_upper, j_diag]))
    #     return A

    # u1 = spsolve(tridiag(a), b)
    # J1 = u1.sum()
    # adj1 = np.array(J1.diff(a)).ravel()
    # pylab.plot(adj1)

    # # finite difference
    # fd = np.zeros(a.size)
    # for i in range(a.size):
    #     a[:] = 1
    #     a[i] = 1 + 1E-6
    #     A = tridiag(a)
    #     du = sp.linalg.spsolve(A._base, b._base) - u._base
    #     fd[i] = du.sum() / 1E-6

    # pylab.plot(fd)

    # print('Adj - Adj1', np.linalg.norm(adj - adj1))
    # print('Adj - fd', np.linalg.norm(adj - fd))
    # print('Adj1 - fd', np.linalg.norm(adj1 - fd))
