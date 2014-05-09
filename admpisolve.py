# solve parallel nonlinear systems, and differentiate through implicit relations
# that are established through nonlinear solvers
# Copyright (C) 2014
# Qiqi Wang  qiqi.wang@gmail.com
# engineer-chaos.blogspot.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#
# The lgmres routine is modified from source code from Scipy 0.13.3
# Copyright (C) 2009, Pauli Virtanen <pav@iki.fi>
# Distributed under the same license as Scipy.


from __future__ import division, print_function, absolute_import

import os
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from scipy.linalg import get_blas_funcs

from mpi4py import MPI
_MPI_COMM = MPI.COMM_WORLD

sys.path.append(os.path.realpath('..')) # for running unittest

from numpad.admpi import *

_DTYPE_IND = sp.csr_matrix((1,1)).nonzero()[0].dtype

# ----------- Jacobian in parallel ------------ #
class MpiJacobian:
    '''
    mode='CSR': f_diff_u[rank_i] = df[my_rank] / du[rank_i]
    mode='CSC': f_diff_u[rank_i] = df[rank_i] / du[my_rank]
    '''
    def __init__(self, f_diff_u, mode):
        self._diff = f_diff_u
        self._find_to_and_from_ranks()

        if mode == 'CSR':
            self._diff = self._csr_to_csc(self._diff)
            self._to_ranks, self._from_ranks = self._from_ranks, self._to_ranks
        else:
            assert mode == 'CSC'

    @staticmethod
    def _csr_to_csc(diff_csr):
        diff_csc = {}
        # diagonal
        my_rank = _MPI_COMM.Get_rank()
        diff_csc[my_rank] = diff_csr[my_rank]

        # send the off-diagonal
        requests = []
        for rank, diff in diff_csr.items():
            if rank != my_rank:
                buf = np.fromstring(pickle.dumps(diff), 'c')
                requests.append(_MPI_COMM.Isend(buf.view(np.int8), rank, 0))
                if my_rank == 2 and rank == 1:
                    open('1.pkl', 'wb').write(pickle.dumps(diff))
                # print('from ', my_rank, ' to ', rank, ' send ', ret)

        num_offdiag_sent = np.zeros((), int)
        _MPI_COMM.Allreduce(np.array(len(requests)), num_offdiag_sent, MPI.SUM)

        # receive the off-diagonals
        num_offdiag_received = np.zeros((), int)  # sum across processes
        num_local_received = 0                    # by the current process

        while num_offdiag_received < num_offdiag_sent:
            status = MPI.Status()
            while _MPI_COMM.Iprobe(MPI.ANY_SOURCE, 0, status):
                buf = np.empty(status.count, 'c')
                _MPI_COMM.Recv(buf.view(np.int8), status.source, 0)
                # print('FROM ', status.source, ' TO ', my_rank, ' recv ', ret)
                diff_csc[status.source] = pickle.loads(buf.tostring())
                num_local_received += 1

            _MPI_COMM.Allreduce(np.array(num_local_received),
                                num_offdiag_received, MPI.SUM)

        for request in requests:
            request.Wait()
        return diff_csc

    def _find_to_and_from_ranks(self):
        # find nonzero rows of off-diagonals
        self._to_ranks = {}
        for rank in self._diff:
            if rank == _MPI_COMM.Get_rank():
                continue
            if self._diff[rank] is 0:
                del self._diff[rank]
                continue
            i_rows = np.unique(self._diff[rank].nonzero()[0])
            if i_rows.size == 0:
                del self._diff[rank]
                continue
            self._to_ranks[rank] = i_rows
            # slice the matrix
            self._diff[rank] = self._diff[rank][i_rows]

        # send the nonzero rows of off-diagonals
        requests = []
        for rank, i_rows in self._to_ranks.items():
            requests.append(_MPI_COMM.Isend(i_rows, rank, 0))

        num_offdiag_sent = np.zeros((), int)
        _MPI_COMM.Allreduce(np.array(len(requests)), num_offdiag_sent, MPI.SUM)

        # receive the off-diagonals
        num_offdiag_received = np.zeros((), int)  # sum across processes
        self._from_ranks = {}

        while num_offdiag_received < num_offdiag_sent:
            status = MPI.Status()
            while _MPI_COMM.Iprobe(MPI.ANY_SOURCE, 0, status):
                assert status.count % _DTYPE_IND.itemsize == 0
                i_rows = np.empty(status.count // _DTYPE_IND.itemsize,
                                  _DTYPE_IND)
                _MPI_COMM.Recv(i_rows, status.source, 0)
                self._from_ranks[status.source] = i_rows

            _MPI_COMM.Allreduce(np.array(len(self._from_ranks)),
                                num_offdiag_received, MPI.SUM)

        for request in requests:
            request.Wait()

    # --------------- matvec and approximate inverse ------------ #

    def matvec(self, du):
        my_rank = _MPI_COMM.Get_rank()
        df = self._diff[my_rank] * du

        if not hasattr(self, '_to_ranks'):
            self._find_to_and_from_ranks()

        requests = []
        for rank in self._to_ranks:
            df_to_remote = self._diff[rank] * du
            requests.append(_MPI_COMM.Isend(df_to_remote, rank, 0))

        for rank, i_rows in self._from_ranks.items():
            df_from_remote = np.empty(i_rows.size)
            _MPI_COMM.Recv(df_from_remote, rank, 0)
            df[i_rows] += df_from_remote

        for request in requests:
            request.Wait()
        return df

    def approx_solve(self, df):
        if not hasattr(self, '_diff_factorized'):
            my_rank = _MPI_COMM.Get_rank()
            self._diff_factorized = \
                    splinalg.factorized(self._diff[my_rank].tocsc())
        return self._diff_factorized(df)


# ----------- iterative linear solver ------------ #

def _norm2(q):
    q = np.asarray(q)
    nrm2_sq = np.array((q**2).sum())
    _MPI_COMM.Allreduce(MPI.IN_PLACE, nrm2_sq, MPI.SUM)
    return float(np.sqrt(nrm2_sq))

def _dot(p, q):
    p, q = np.asarray(p), np.asarray(q)
    assert p.dtype == q.dtype
    p_dot_q = np.array(np.dot(p, q))
    _MPI_COMM.Allreduce(MPI.IN_PLACE, p_dot_q, MPI.SUM)
    return float(p_dot_q)

def _lgmres(A, b, x0=None, tol=1e-5, maxiter=1000, M=None, callback=None,
           inner_m=30, outer_k=3, outer_v=None, store_outer_Av=True):
    """
    Solve a matrix equation using the LGMRES algorithm.

    The LGMRES algorithm [BJM]_ [BPh]_ is designed to avoid some problems
    in the convergence in restarted GMRES, and often converges in fewer
    iterations.

    Parameters
    ----------
    A : {sparse matrix, dense matrix, LinearOperator}
        The real or complex N-by-N matrix of the linear system.
    b : {array, matrix}
        Right hand side of the linear system. Has shape (N,) or (N,1).
    x0  : {array, matrix}
        Starting guess for the solution.
    tol : float
        Tolerance to achieve. The algorithm terminates when either the relative
        or the absolute residual is below `tol`.
    maxiter : int
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    M : {sparse matrix, dense matrix, LinearOperator}
        Preconditioner for A.  The preconditioner should approximate the
        inverse of A.  Effective preconditioning dramatically improves the
        rate of convergence, which implies that fewer iterations are needed
        to reach a given error tolerance.
    callback : function
        User-supplied function to call after each iteration.  It is called
        as callback(xk), where xk is the current solution vector.
    inner_m : int, optional
        Number of inner GMRES iterations per each outer iteration.
    outer_k : int, optional
        Number of vectors to carry between inner GMRES iterations.
        According to [BJM]_, good values are in the range of 1...3.
        However, note that if you want to use the additional vectors to
        accelerate solving multiple similar problems, larger values may
        be beneficial.
    outer_v : list of tuples, optional
        List containing tuples ``(v, Av)`` of vectors and corresponding
        matrix-vector products, used to augment the Krylov subspace, and
        carried between inner GMRES iterations. The element ``Av`` can
        be `None` if the matrix-vector product should be re-evaluated.
        This parameter is modified in-place by `lgmres`, and can be used
        to pass "guess" vectors in and out of the algorithm when solving
        similar problems.
    store_outer_Av : bool, optional
        Whether LGMRES should store also A*v in addition to vectors `v`
        in the `outer_v` list. Default is True.

    Returns
    -------
    x : array or matrix
        The converged solution.
    info : int
        Provides convergence information:

            - 0  : successful exit
            - >0 : convergence to tolerance not achieved, number of iterations
            - <0 : illegal input or breakdown

    Notes
    -----
    The LGMRES algorithm [BJM]_ [BPh]_ is designed to avoid the
    slowing of convergence in restarted GMRES, due to alternating
    residual vectors. Typically, it often outperforms GMRES(m) of
    comparable memory requirements by some measure, or at least is not
    much worse.

    Another advantage in this algorithm is that you can supply it with
    'guess' vectors in the `outer_v` argument that augment the Krylov
    subspace. If the solution lies close to the span of these vectors,
    the algorithm converges faster. This can be useful if several very
    similar matrices need to be inverted one after another, such as in
    Newton-Krylov iteration where the Jacobian matrix often changes
    little in the nonlinear steps.

    References
    ----------
    .. [BJM] A.H. Baker and E.R. Jessup and T. Manteuffel,
             SIAM J. Matrix Anal. Appl. 26, 962 (2005).
    .. [BPh] A.H. Baker, PhD thesis, University of Colorado (2003).
             http://amath.colorado.edu/activities/thesis/allisonb/Thesis.ps

    """
    from scipy.linalg.basic import lstsq
    # A,M,x,b,postprocess = make_system(A,M,x0,b)
    x = x0

    if not np.isfinite(b).all():
        raise ValueError("RHS must contain only finite numbers")

    matvec = A
    psolve = M

    if outer_v is None:
        outer_v = []

    axpy, scal = None, None

    b_norm = _norm2(b)
    if b_norm == 0:
        b_norm = 1

    for k_outer in range(maxiter):
        r_outer = matvec(x) - b

        # -- callback
        if callback is not None:
            callback(x)

        # -- determine input type routines
        if axpy is None:
            # if np.iscomplexobj(r_outer) and not np.iscomplexobj(x):
            #     x = x.astype(r_outer.dtype)
            # axpy, dot, scal = get_blas_funcs(['axpy', 'dot', 'scal'],
            #                                   (x, r_outer))
            axpy, scal = get_blas_funcs(['axpy', 'scal'], (x, r_outer))

        # -- check stopping condition
        r_norm = _norm2(r_outer)
        if (_MPI_COMM.Get_rank() == 0):
            print('        lgmres', k_outer, r_norm)
        if r_norm < tol * b_norm or r_norm < tol:
            break

        # -- inner LGMRES iteration
        vs0 = -psolve(r_outer)
        inner_res_0 = _norm2(vs0)

        if inner_res_0 == 0:
            rnorm = _norm2(r_outer)
            raise RuntimeError("Preconditioner returned a zero vector; "
                               "|v| ~ %.1g, |M v| = 0" % rnorm)

        vs0 = scal(1.0/inner_res_0, vs0)
        hs = []
        vs = [vs0]
        ws = []
        y = None

        for j in range(1, 1 + inner_m + len(outer_v)):
            # -- Arnoldi process:
            #
            #    Build an orthonormal basis V and matrices W and H such that
            #        A W = V H
            #    Columns of W, V, and H are stored in `ws`, `vs` and `hs`.
            #
            #    The first column of V is always the residual vector, `vs0`;
            #    V has *one more column* than the other of the three matrices.
            #
            #    The other columns in V are built by feeding in, one
            #    by one, some vectors `z` and orthonormalizing them
            #    against the basis so far. The trick here is to
            #    feed in first some augmentation vectors, before
            #    starting to construct the Krylov basis on `v0`.
            #
            #    It was shown in [BJM]_ that a good choice (the LGMRES choice)
            #    for these augmentation vectors are the `dx` vectors obtained
            #    from a couple of the previous restart cycles.
            #
            #    Note especially that while `vs0` is always the first
            #    column in V, there is no reason why it should also be
            #    the first column in W. (In fact, below `vs0` comes in
            #    W only after the augmentation vectors.)
            #
            #    The rest of the algorithm then goes as in GMRES, one
            #    solves a minimization problem in the smaller subspace
            #    spanned by W (range) and V (image).
            #
            #    XXX: Below, I'm lazy and use `lstsq` to solve the
            #    small least squares problem. Performance-wise, this
            #    is in practice acceptable, but it could be nice to do
            #    it on the fly with Givens etc.
            #

            #     ++ evaluate
            v_new = None
            if j < len(outer_v) + 1:
                z, v_new = outer_v[j-1]
            elif j == len(outer_v) + 1:
                z = vs0
            else:
                z = vs[-1]

            if v_new is None:
                v_new = psolve(matvec(z))
            else:
                # Note: v_new is modified in-place below. Must make a
                # copy to ensure that the outer_v vectors are not
                # clobbered.
                v_new = v_new.copy()

            #     ++ orthogonalize
            hcur = []
            for v in vs:
                alpha = _dot(v, v_new)
                hcur.append(alpha)
                v_new = axpy(v, v_new, v.shape[0], -alpha)  # v_new -= alpha*v
            hcur.append(_norm2(v_new))

            if hcur[-1] == 0:
                # Exact solution found; bail out.
                # Zero basis vector (v_new) in the least-squares problem
                # does no harm, so we can just use the same code as usually;
                # it will give zero (inner) residual as a result.
                bailout = True
            else:
                bailout = False
                v_new = scal(1.0/hcur[-1], v_new)

            vs.append(v_new)
            hs.append(hcur)
            ws.append(z)

            # XXX: Ugly: should implement the GMRES iteration properly,
            #      with Givens rotations and not using lstsq. Instead, we
            #      spare some work by solving the LSQ problem only every 5
            #      iterations.
            if not bailout and j % 5 != 1 and j < inner_m + len(outer_v) - 1:
                continue

            # -- GMRES optimization problem
            hess = np.zeros((j+1, j), x.dtype)
            e1 = np.zeros((j+1,), x.dtype)
            e1[0] = inner_res_0
            for q in range(j):
                hess[:(q+2),q] = hs[q]

            y, resids, rank, s = lstsq(hess, e1)
            inner_res = _norm2(np.dot(hess, y) - e1)

            # -- check for termination
            if inner_res < tol * inner_res_0:
                break

        # -- GMRES terminated: eval solution
        dx = ws[0]*y[0]
        for w, yc in zip(ws[1:], y[1:]):
            dx = axpy(w, dx, dx.shape[0], yc)  # dx += w*yc

        # -- Store LGMRES augmentation vectors
        nx = _norm2(dx)
        if store_outer_Av:
            q = np.dot(hess, y)
            ax = vs[0]*q[0]
            for v, qc in zip(vs[1:], q[1:]):
                ax = axpy(v, ax, ax.shape[0], qc)
            outer_v.append((dx/nx, ax/nx))
        else:
            outer_v.append((dx/nx, None))

        # -- Retain only a finite number of augmentation vectors
        while len(outer_v) > outer_k:
            del outer_v[0]

        # -- Apply step
        x += dx
    else:
        # didn't converge ...
        return x, maxiter

    return x, 0


# ----------- nonlinear solver ------------ #
def solve_mpi(func, u0, args=(), kargs={},
          max_iter=10, abs_tol=1E-6, rel_tol=1E-6, verbose=True):
    pass


comm_size = np.sqrt(COMM_WORLD.Get_size()), np.sqrt(COMM_WORLD.Get_size())
assert np.prod(comm_size) == COMM_WORLD.Get_size()

def i_rank():
    return COMM_WORLD.Get_rank() // comm_size[1]

def j_rank():
    return COMM_WORLD.Get_rank() % comm_size[1]

def rank(i, j):
    return i * comm_size[1] + j

from numpad import *
M, N = 50, 50
u = zeros((N, M))
f = ones((N, M))
dx = 1. / (N * comm_size[1] + 1)
dy = 1. / (M * comm_size[0] + 1)

def residual(u):
    u_up = zeros(N + 2)
    u_down = zeros(N + 2)
    u_right = zeros(M)
    u_left = zeros(M)
    
    if i_rank() > 0:
        u_down[1:-1] = u[0,:]
        COMM_WORLD.Send(u_down[1:-1], rank(i_rank() - 1, j_rank()))
    if i_rank() < comm_size[0] - 1:
        COMM_WORLD.Recv(u_down[1:-1], rank(i_rank() + 1, j_rank()))
    else:
        u_down[:] = 0
    
    if i_rank() < comm_size[0] - 1:
        u_up[1:-1] = u[-1,:]
        COMM_WORLD.Send(u_up[1:-1], rank(i_rank() + 1, j_rank()))
    if i_rank() > 0:
        COMM_WORLD.Recv(u_up[1:-1], rank(i_rank() - 1, j_rank()))
    else:
        u_up[:] = 0

    if j_rank() > 0:
        u_right[:] = u[:,0]
        COMM_WORLD.Send(u_right, rank(i_rank(), j_rank() - 1))
    if j_rank() < comm_size[1] - 1:
        COMM_WORLD.Recv(u_right, rank(i_rank(), j_rank() + 1))
    else:
        u_right[:] = 0
    
    if j_rank() < comm_size[1] - 1:
        u_left[:] = u[:,-1]
        COMM_WORLD.Send(u_left, rank(i_rank(), j_rank() + 1))
    if j_rank() > 0:
        COMM_WORLD.Recv(u_left, rank(i_rank(), j_rank() - 1))
    else:
        u_left[:] = 0

    u_ext = hstack([u_left[:,np.newaxis], u, u_right[:,np.newaxis]])
    u_ext = vstack([u_up[np.newaxis,:], u_ext, u_down[np.newaxis,:]])
    return (u_ext[1:-1,2:] + u_ext[1:-1,:-2] - 2 * u_ext[1:-1,1:-1]) / dx**2 \
         + (u_ext[2:,1:-1] + u_ext[:-2,1:-1] - 2 * u_ext[1:-1,1:-1]) / dy**2 \
         - f


r = residual(u)
r_diff_u = diff_mpi(r, u, 'tangent')

J = MpiJacobian(r_diff_u, 'CSR')

uu, _ = _lgmres(J.matvec, r._value.ravel(), x0=u._value.ravel(), tol=1e-5, maxiter=1000,
        M=J.approx_solve, callback=None,
        inner_m=30, outer_k=3, outer_v=None, store_outer_Av=True)

if COMM_WORLD.Get_rank() > 0:
    COMM_WORLD.Send(adarray(uu), 0)
else:
    uAll = zeros(N * M * COMM_WORLD.Get_size())
    uAll[:N*M] = uu
    for i in range(1, COMM_WORLD.Get_size()):
        COMM_WORLD.Recv(uAll[N*M*i:N*M*(i+1)], i)
    import pylab
    uAll = uAll.reshape((comm_size[0], comm_size[1], M, N))
    uAll = uAll.transpose([0,2,1,3])
    uAll = uAll.reshape([comm_size[0] * N, comm_size[0] * M])
    pylab.plot(value(uAll).T)
    pylab.show()
