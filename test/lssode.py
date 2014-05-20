# Copyright Qiqi Wang (qiqi@mit.edu) 2013
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

"""This module contains tools for performing tangnet sensitivity analysis
and adjoint sensitivity analysis.  The details are described in our paper
"Sensitivity computation of periodic and chaotic limit cycle oscillations"
at http://arxiv.org/abs/1204.0159

User should define two bi-variate functions, f and J

f(u, s) defines a dynamical system du/dt = f(u,s) parameterized by s
        inputs:
        u: size (m,) or size (N,m). It's the state of the m-degree-of-freedom
           dynamical system
        s: parameter of the dynamical system.
           Tangent sensitivity analysis: s must be a scalar.
           Adjoint sensitivity analysis: s may be a scalar or vector.
        return: du/dt, should be the same size as the state u.
                if u.shape == (m,): return a shape (m,) array
                if u.shape == (N,m): return a shape (N,m) array

J(u, s) defines the objective function, whose ergodic long time average
        is the quantity of interest.
        inputs: Same as in f(u,s)
        return: instantaneous objective function to be time averaged.
                Tangent sensitivity analysis:
                    J may return a scalar (single objectives)
                              or a vector (n objectives).
                    if u.shape == (m,): return a scalar or vector of shape (n,)
                    if u.shape == (N,m): return a vector of shape (N,)
                                         or vector of shape (N,n)
                Adjoint sensitivity analysis:
                    J must return a scalar (single objective).
                    if u.shape == (m,): return a scalar
                    if u.shape == (N,m): return a vector of shape (N,)

Using tangent sensitivity analysis:
        u0 = rand(m)      # initial condition of m-degree-of-freedom system
        t = linspace(T0, T1, N)    # 0-T0 is spin up time (starting from u0).
        tan = Tangent(f, u0, s, t)
        dJds = tan.dJds(J)
        # you can use the same "tan" for more "J"s ...

Using adjoint sensitivity analysis:
        adj = Adjoint(f, u0, s, t, J)
        dJds = adj.dJds()
        # you can use the same "adj" for more "s"s
        #     via adj.dJds(dfds, dJds)... See doc for the Adjoint class

Using nonlinear LSS solver:
        u0 = rand(m)      # initial condition of m-degree-of-freedom system
        t = linspace(T0, T1, N)    # 0-T0 is spin up time (starting from u0).
        solver = lssSolver(f, u0, s0, t)
        # (solver.t, solver.u) is the solution of initial value problem at s0
        solver.lss(s1)
        # (solver.t, solver.u) is the solution of a LSS problem at s
"""
import sys
from scipy.integrate import odeint

sys.path.append('../..')
sys.path.append('..')
import numpad as np
from numpad import sparse
from numpad.adsparse import spsolve
# import numpy as np
# from scipy import sparse
# from scipy.sparse.linalg import spsolve


import resource
def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF)
    return '''%s: usertime=%s systime=%s mem=%s mb
           '''%(point,usage[0],usage[1],
                (usage[2]*resource.getpagesize())/1000000.0 )

__all__ = ["ddu", "dds", "set_fd_step", "Tangent", "Adjoint", "lssSolver"]


def _diag(a):
    """Construct a block diagonal sparse matrix, A[i,:,:] is the ith block"""
    assert a.ndim == 1
    n = a.size
    return sparse.csr_matrix((a, np.arange(n), np.arange(n+1)))

def _block_diag(A):
    """Construct a block diagonal sparse matrix, A[i,:,:] is the ith block"""
    assert A.ndim == 3
    n = A.shape[0]
    return sparse.bsr_matrix((A, np.arange(n), np.arange(n+1)))


EPS = 1E-7

def set_fd_step(eps):
    """Set step size in ddu and dds classess.
    set eps=1E-30j for complex derivative method."""
    assert isinstance(eps, (float, complex))
    global EPS
    EPS = eps


class ddu(object):
    """Partial derivative of a bivariate function f(u,s)
    with respect its FIRST argument u

    Usage: print(ddu(f)(u,s))
    Or: dfdu = ddu(f)
        print(dfdu(u,s))
    """
    def __init__(self, f):
        self.f = f

    def __call__(self, u, s):
        global EPS
        f0 = self.f(u, s)
        assert f0.shape[0] == u.shape[0]
        N = f0.shape[0]
        n, m = f0.size / N, u.shape[1]
        dfdu = np.zeros( (N, n, m) )
        u = np.asarray(u, type(EPS))
        s = np.asarray(s, type(EPS))
        for i in range(m):
            u[:,i] += EPS
            fp = self.f(u, s).copy()
            u[:,i] -= EPS * 2
            fm = self.f(u, s).copy()
            u[:,i] += EPS
            dfdu[:,:,i] = ((fp - fm).reshape([N, n]) / (2 * EPS))
        return dfdu


class dds(object):
    """Partial derivative of a bivariate function f(u,s)
    with respect its SECOND argument s

    Usage: print(dds(f)(u,s))
    Or: dfds = dds(f)
        print(dfds(u,s))
    """
    def __init__(self, f):
        self.f = f

    def __call__(self, u, s):
        global EPS
        f0 = self.f(u, s)
        assert f0.shape[0] == u.shape[0]
        N = f0.shape[0]
        n, m = f0.size / N, s.size
        dfds = np.zeros( (N, n, m) )
        u = np.asarray(u, type(EPS))
        s = np.asarray(s, type(EPS))
        for i in range(m):
            s[i] += EPS
            fp = self.f(u, s).copy()
            s[i] -= EPS * 2
            fm = self.f(u, s).copy()
            s[i] += EPS
            dfds[:,:,i] = ((fp - fm).reshape([N, n]) / (2 * EPS))
        return dfds


class LSS(object):
    """
    Base class for both tangent and adjoint sensitivity analysis
    During __init__, a trajectory is computed,
    and the matrices used for both tangent and adjoint are built
    """
    def __init__(self, f, u0, s, t, dfdu=None):
        self.f = f
        self.t = np.array(t, float).copy()
        self.s = np.array(s, float).copy()

        if self.s.ndim == 0:
            self.s = self.s[np.newaxis]

        if dfdu is None:
            dfdu = ddu(f)
        self.dfdu = dfdu

        u0 = np.array(u0, float)
        if u0.ndim == 1:
            # run up to t[0]
            f = lambda u, t : self.f(u, s)
            assert t[0] >= 0 and t.size > 1
            N0 = int(t[0] / (t[-1] - t[0]) * t.size)
            u0 = odeint(f, u0, np.linspace(0, t[0], N0+1))[-1]

            # compute a trajectory
            self.u = np.array(odeint(f, u0, t - t[0]))
        else:
            assert (u0.shape[0],) == t.shape
            self.u = u0.copy()

        self.dt = self.t[1:] - self.t[:-1]
        self.uMid = 0.5 * (self.u[1:] + self.u[:-1])
        self.dudt = (self.u[1:] - self.u[:-1]) / self.dt[:,np.newaxis]

    def Schur(self, alpha):
        """
        Builds the Schur complement of the KKT system'
        Also build B: the block-bidiagonal matrix,
               and E: the dudt matrix
        """
        N, m = self.u.shape[0] - 1, self.u.shape[1]

        halfJ = 0.5 * self.dfdu(self.uMid, self.s)
        eyeDt = np.eye(m,m) / self.dt[:,np.newaxis,np.newaxis]

        E = -eyeDt - halfJ
        f = self.dudt
        G = +eyeDt - halfJ
        self.E, self.G = E, G

        def block_ij_to_element_ij(i, j, m):
            i_addition = np.arange(m)[:,np.newaxis] + np.zeros([m,m], int)
            j_addition = np.arange(m)[np.newaxis,:] + np.zeros([m,m], int)
            i = i[:,np.newaxis,np.newaxis] * m + i_addition
            j = j[:,np.newaxis,np.newaxis] * m + j_addition
            return i, j

        # construct B * B.T
        diag_data = (E[:,:,np.newaxis,:] * E[:,np.newaxis,:,:]).sum(3) \
                  + f[:,:,np.newaxis] * f[:,np.newaxis,:] / alpha**2 \
                  + (G[:,:,np.newaxis,:] * G[:,np.newaxis,:,:]).sum(3)

        upper_data = (G[:-1,:,np.newaxis,:] * E[1:,np.newaxis,:,:]).sum(3)
        lower_data = upper_data.transpose([0,2,1])

        diag_i = np.arange(diag_data.shape[0])
        diag_j = diag_i
        upper_i = np.arange(diag_data.shape[0] - 1)
        upper_j = upper_i + 1
        lower_i, lower_j = upper_j, upper_i

        diag_i, diag_j = block_ij_to_element_ij(diag_i, diag_j, m)
        upper_i, upper_j = block_ij_to_element_ij(upper_i, upper_j, m)
        lower_i, lower_j = block_ij_to_element_ij(lower_i, lower_j, m)

        data = np.hstack([np.ravel(diag_data), np.ravel(upper_data), np.ravel(lower_data)])
        i = np.hstack([np.ravel(diag_i), np.ravel(upper_i), np.ravel(lower_i)])
        j = np.hstack([np.ravel(diag_j), np.ravel(upper_j), np.ravel(lower_j)])

        return sparse.csr_matrix((data, (i, j)))

    def evaluate(self, J):
        """Evaluate a time averaged objective function"""
        return J(self.u, self.s).mean(0)


class Tangent(LSS):
    """
    Tagent(f, u0, s, t, dfds=None, dfdu=None, alpha=10)
    f: governing equation du/dt = f(u, s)
    u0: initial condition (1d array) or the entire trajectory (2d array)
    s: parameter
    t: time (1d array).  t[0] is run up time from initial condition.
    dfds and dfdu is computed from f if left undefined.
    alpha: weight of the time dilation term in LSS.
    """
    def __init__(self, f, u0, s, t, dfds=None, dfdu=None, alpha=10):
        LSS.__init__(self, f, u0, s, t, dfdu)

        Smat = self.Schur(alpha)

        if dfds is None:
            dfds = dds(f)
        b = dfds(self.uMid, self.s)
        assert b.size == Smat.shape[0]

        w = spsolve(Smat, np.ravel(b))
        v = self.wBinv * (self.B.T * w)

        self.v = v.reshape(self.u.shape)
        self.eta = self.wEinv * (self.E.T * w)

    def dJds(self, J, T0skip=0, T1skip=0):
        """Evaluate the derivative of the time averaged objective function to s
        """
        pJpu, pJps = ddu(J), dds(J)

        n0 = (self.t < self.t[0] + T0skip).sum()
        n1 = (self.t <= self.t[-1] - T1skip).sum()
        assert n0 < n1

        u, v = self.u[n0:n1], self.v[n0:n1]
        uMid, eta = self.uMid[n0:n1-1], self.eta[n0:n1-1]

        J0 = J(uMid, self.s)
        J0 = J0.reshape([uMid.shape[0], -1])

        grad1 = (pJpu(u, self.s) * v[:,np.newaxis,:]).sum(2).mean(0) \
              - (eta[:,np.newaxis] * (J0 - J0.mean(0))).mean(0)

        grad2 = pJps(uMid, self.s)[:,:,0].mean(0)
        return np.ravel(grad1 + grad2)


class Adjoint(LSS):
    """
    Adjoint(f, u0, s, t, J, dJdu=None, dfdu=None, alpha=10)
    f: governing equation du/dt = f(u, s)
    u0: initial condition (1d array) or the entire trajectory (2d array)
    s: parameter
    t: time (1d array).  t[0] is run up time from initial condition.
    J: objective function. QoI = mean(J(u))
    dJdu and dfdu is computed from f if left undefined.
    alpha: weight of the time dilation term in LSS.
    """
    def __init__(self, f, u0, s, t, J, dJdu=None, dfdu=None, alpha=10):
        LSS.__init__(self, f, u0, s, t, dfdu)

        Smat = self.Schur(alpha)

        J0 = J(self.uMid, self.s)
        assert J0.ndim == 1
        h = -(J0 - J0.mean()) / J0.size            # multiplier on eta

        if dJdu is None:
            dJdu = ddu(J)
        g = dJdu(self.u, self.s) / self.u.shape[0]  # multiplier on v
        assert g.size == self.u.size

        b = self.E * (self.wEinv * h) + self.B * (self.wBinv * np.ravel(g))
        wa = spsolve(Smat, b)

        self.wa = wa.reshape(self.uMid.shape)
        self.J, self.dJdu = J, dJdu

    def evaluate(self):
        """Evaluate the time averaged objective function"""
        # return self.J(self.u, self.s).mean(0)
        return LSS.evaluate(self, self.J)

    def dJds(self, dfds=None, dJds=None, T0skip=0, T1skip=0):
        """Evaluate the derivative of the time averaged objective function to s
        """
        if dfds is None:
            dfds = dds(self.f)
        if dJds is None:
            dJds = dds(self.J)

        n0 = (self.t < self.t[0] + T0skip).sum()
        n1 = (self.t <= self.t[-1] - T1skip).sum()

        uMid, wa = self.uMid[n0:n1-1], self.wa[n0:n1-1]

        prod = self.wa[:,:,np.newaxis] * dfds(self.uMid, self.s)
        grad1 = prod.sum(0).sum(0)
        grad2 = dJds(self.uMid, self.s).mean(0)
        return np.ravel(grad1 + grad2)


class lssSolver(LSS):
    """
    lssSolver(f, u0, s, t, dfds=None, dfdu=None, alpha=10)
    f: governing equation du/dt = f(u, s)
    u0: initial condition (1d array) or the entire trajectory (2d array)
    s: parameter
    t: time (1d array).  t[0] is run up time from initial condition.
    dfds and dfdu is computed from f if left undefined.
    alpha: weight of the time dilation term in LSS.
    """
    def __init__(self, f, u0, s, t, dfdu=None, alpha=10):
        LSS.__init__(self, f, u0, s, t, dfdu)
        self.alpha = alpha

    def lss(self, s, maxIter=8, atol=1E-7, rtol=1E-8, disp=False):
        """Compute a new nonlinear solution at a different s.
        This one becomes the reference solution for the next call"""
        
        # print(using("Start lss solver "))

        N, m = self.u.shape[0] - 1, self.u.shape[1]

        Smat = self.Schur(self.alpha)

        s = np.array(s, float).copy()
        if s.ndim == 0:
            s = s[np.newaxis]
        assert s.shape == self.s.shape
        self.s = s

        # compute initial matrix and right hand side
        b = self.dudt - self.f(self.uMid, s)
        norm_b0 = np.sqrt((np.ravel(b)**2).sum())

        Smat = self.Schur(self.alpha)

        u_adj = np.zeros(self.u.shape)
        eta_adj = np.zeros(b.shape)

        for iNewton in range(maxIter):
            
            # print(using("Start newton iter "+str(iNewton)))

            # solve
            w = spsolve(Smat, np.ravel(b))
            w = w.reshape([-1, m])
            GTw = (self.G * w[:,:,np.newaxis]).sum(1)
            ETw = (self.E * w[:,:,np.newaxis]).sum(1)
            v = -np.vstack([np.zeros([1,m]), GTw]) \
                -np.vstack([ETw, np.zeros([1,m])])

            eta = -(self.dudt * w).sum(1) / self.alpha**2

            # update solution and dt
            u = self.u
            self.u = self.u + v
            self.dt = self.dt/np.exp(eta)

            # evaluate costfunction
            TARGET = 2.8
            J=(self.u[:,1]**8).mean(0)
            J=J**(1./8)
            J=1./2*(J-TARGET)**2
            print('J '+str(J))

            #testing derivatives
            #u_adj = u_adj + array(J.diff(u)) # - dot(v.diff(u),u_adj)
#            print((v * u_adj).sum())
#            print(((v * u_adj).sum().diff(u)).reshape(u_adj.shape))
#            J0 = np.array(J.diff(self.u).todense()).reshape(u_adj.shape)
#            print(J0[0,1])

            # update adjoint
#            u_update= np.array(J.diff(self.u).todense()).reshape(u_adj.shape) \
#                          - np.array(((v * u_adj).sum()).diff(u)).reshape(u_adj.shape)
#            u_adj = u_adj + np.array(J.diff(self.u).todense()).reshape(u_adj.shape) \
#                          - np.array(((v * u_adj).sum()).diff(u)).reshape(u_adj.shape)

#            u_adj=u_adj+u_update
#            print(u_adj[0,0])
#            norm=np.sqrt((np.ravel(u_update)**2).sum())
#            print(norm)
#            stop

            

            # compute gradient
#            grad = J.diff(s) - dot(v.diff(s),u_adj)

            self.uMid = 0.5 * (self.u[1:] + self.u[:-1])
            self.dudt = (self.u[1:] - self.u[:-1]) / self.dt[:,np.newaxis]
            print(using("after u update iter "+str(iNewton)))
            self.t[1:] = self.t[0] + np.cumsum(self.dt)
#            print(using("after cumsum iter "+str(iNewton)))
            
            # recompute residual
            b = self.dudt - self.f(self.uMid, s)
            norm_b = np.sqrt((np.ravel(b)**2).sum())
            if disp:
                print('iteration, norm_b, norm_b0 ', iNewton, norm_b, norm_b0)
#            if norm_b < atol or norm_b < rtol * norm_b0:
#                return self.t, self.u

            # recompute matrix
            Smat = self.Schur(self.alpha)

        # did not meet tolerance, error message
        print('lssSolve: Newton solver did not converge in {0} iterations')
