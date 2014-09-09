# admpi.py performs AD in parallel
# of one state with respect to another
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

from __future__ import division, print_function, absolute_import

import os
import sys
import unittest
import weakref
import pickle
import numpy as np
import scipy.sparse as sp

from mpi4py import MPI
_MPI_COMM = MPI.COMM_WORLD

sys.path.append(os.path.realpath('..')) # for running unittest

from numpad.adstate import *
from numpad.adarray import adarray


# ----------- Subclassing IntermediateState ----------- #

class MpiSendState(IntermediateState):
    '''
    '''
    def __init__(self, prev_state, dest, tag):
        host = prev_state.host()
        IntermediateState.__init__(self, host, prev_state, 1, None)

        self.dest = dest
        self.tag = tag

        self.cls_send_states[self._state_id] = self 

    def after_diff_tangent(self, self_diff_u):
        '''
        send self_diff_u to remote after diff_tangent computes it
        '''
        buf = np.fromstring(pickle.dumps(self_diff_u), 'c')
        _MPI_COMM.Send((buf, MPI.BYTE), self.dest, self.tag)

    # centralized management of all MpISendState objects
    cls_send_states = {}  # strong refs

    cls_state = {'count_activated': 0}

    @staticmethod
    def count_activated():
        return MpiSendState.cls_state['count_activated']

    @staticmethod
    def start_waiting():
        MpiSendState.cls_state['count_activated'] = 0

    @staticmethod
    def newly_activated():
        status = MPI.Status()
        while _MPI_COMM.Iprobe(MPI.ANY_SOURCE, MPI.ANY_TAG, status):
            assert status.count == np.dtype(int).itemsize
            state_id = np.empty((), int)
            _MPI_COMM.Recv(state_id, status.source, status.tag)
            state_id = int(state_id)

            assert state_id in MpiSendState.cls_send_states
            yield MpiSendState.cls_send_states[state_id]
            MpiSendState.cls_state['count_activated'] += 1


class MpiRecvState(IntermediateState):
    '''
    '''
    def __init__(self, prev_state, source, tag, send_state_id):
        host = prev_state.host()
        IntermediateState.__init__(self, host, prev_state, 0, None)

        self.source = source
        self.tag = tag
        self.send_state_id = send_state_id

        self.cls_recv_states[(source, send_state_id)] = weakref.ref(self)

    def after_diff_tangent(self, self_diff_u):
        '''
        add the remote self_diff_u value to the local value computed
        by diff_tangent
        '''
        status = MPI.Status()
        _MPI_COMM.Probe(self.source, self.tag, status)
        buf = np.empty(status.count, 'c')
        _MPI_COMM.Recv((buf, MPI.BYTE), self.source, self.tag)
        self_diff_u_remote = pickle.loads(buf.tostring())

        for rank, diff_remote in self_diff_u_remote.items():
            if rank in self_diff_u:
                self_diff_u[rank] = _add_ops(self_diff_u[rank], diff_remote)
            else:
                self_diff_u[rank] = diff_remote

    def activate_remote(self):
        '''
        In the backwards sweep of tangent differentiation, inform
        that the remote MpiSendState that f depends on it
        '''
        send_state_id = np.array(self.send_state_id, int)
        _MPI_COMM.Send(send_state_id, self.source, self.tag)

    # centralized management of all MpIRecvState objects
    cls_recv_states = {}  # weak refs


# ----------- Overloading MPI calls ------------ #

class COMM_WORLD:
    '''
    Emulating mpi4py.MPI.COMM_WORLD
    '''
    @staticmethod
    def Get_rank():
        return _MPI_COMM.Get_rank()

    @staticmethod
    def Get_size():
        return _MPI_COMM.Get_size()

    @staticmethod
    def Barrier():
        return _MPI_COMM.Barrier()

    # ------------- interesting stuff begins here -------------- #

    @staticmethod
    def Send(buf, dest, tag=0):
        assert isinstance(buf, adarray)
        # 1. send the data
        _MPI_COMM.Send(buf._value, dest, tag)
        # 2. create the SendState
        buf._current_state = MpiSendState(buf._current_state, dest, tag)
        # 2. send the state_id of the SendState we just created
        state_id = np.array(buf._current_state._state_id, dtype=int)
        _MPI_COMM.Send(state_id, dest, tag)

    @staticmethod
    def Recv(buf, source, tag=0):
        assert isinstance(buf, adarray)
        # 1. recv the data
        _MPI_COMM.Recv(buf._value, source, tag)
        # 2. recv the state_id of the matching SendState in the source process
        send_state_id = np.empty((), int)
        _MPI_COMM.Recv(send_state_id, source, tag)
        # 3. create the RecvState
        buf._current_state = MpiRecvState(buf._current_state, source, tag,
                                          int(send_state_id))


# ----------- tangent and adjoint differentiation in parallel ------------ #
#                   in the IntermediateState (low) level

def diff_tangent_mpi(f, u):
    '''
    Computes derivative of f with respect to u, by accumulating Jacobian
    forward, i.e., starting from u
    Must be call from all MPI processes collectively
    '''
    my_rank = _MPI_COMM.Get_rank()
    diff_u = {}   # diff_u[state] = {rank_i: state_diff_u_i, ...}

    # backward sweep, populate diff_u with keys that contain all states
    # that f (directly or indirectly) depends on
    to_visit = [f]
    count_activating = 0
    MpiSendState.start_waiting()

    while True:
        while to_visit:
            state = to_visit.pop(0)
            if state not in diff_u:
                diff_u[state] = {}  # see diff_u
                to_visit.extend(state.froms())

                if isinstance(state, MpiRecvState):
                    state.activate_remote()
                    count_activating += 1

        imbalance = np.array(count_activating - MpiSendState.count_activated())
        _MPI_COMM.Allreduce(MPI.IN_PLACE, imbalance, MPI.SUM)
        if imbalance == 0:
            break
        else:
            assert imbalance > 0
            to_visit.extend(MpiSendState.newly_activated())

    # forward sweep
    for state in sorted(diff_u):  # iterate from earliest state
        if state is u:            # found u in the graph
            diff_u[state] = {my_rank: sp.eye(u.size, u.size)}
        else:                     # compute derivative from its dependees
            ranks = set().union(*(diff_u[s].keys() for s in state.froms()))
            diff_u[state] = {}
            for rank in ranks:
                dependees_diff_u = (diff_u[s].setdefault(rank, 0)
                                    for s in state.froms())
                diff_u[state][rank] = state.diff_tangent(dependees_diff_u)

            if hasattr(state, 'after_diff_tangent'):
                state.after_diff_tangent(diff_u[state])

    # the diagonal block is responsible for knowing the shape
    if my_rank not in diff_u[f] or diff_u[f][my_rank] is 0:
        diff_u[f][my_rank] = sp.csr_matrix((f.size, u.size))

    return diff_u[f]

# --------------- differentiation of adarray (high level) --------------- #

def diff_mpi(f, u, mode='auto'):
    if mode == 'tangent':
        derivative = diff_tangent_mpi(f._current_state, u._initial_state)
    else:
        raise NotImplementedError()

    return derivative


# =========================================================== #
#                                                             #
#                         unittests                           #
#                                                             #
# =========================================================== #

class _SimpleSendRecv(unittest.TestCase):
    def testSendRecv(self):
        self.assertGreater(COMM_WORLD.Get_size(), 1)

        N = 10000
        u = ones(N)
        if COMM_WORLD.Get_rank() == 0:
            COMM_WORLD.Recv(u, 1)
        elif COMM_WORLD.Get_rank() == 1:
            COMM_WORLD.Send(u, 0)
        f = u * u
        f_diff_u = diff_mpi(f, u, mode='tangent')

        # check result
        if COMM_WORLD.Get_rank() == 0:
            nz_rank = 1
        else:
            nz_rank = COMM_WORLD.Get_rank()

        discrepancy = f_diff_u[nz_rank] - 2 * sp.eye(N, N)
        if discrepancy.nnz > 0:
            self.assertAlmostEqual(0, np.abs(discrepancy.data).max())

        for rank in f_diff_u:
            if rank != nz_rank:
                if f_diff_u[rank] is not 0 and f_diff_u[rank].nnz > 0:
                    self.assertAlmostEqual(0, np.abs(f_diff_u[rank].data).max())

    def testPoisson1DResidual(self):
        N = 10000
        u = random(N)
        dx = 1. / (N * COMM_WORLD.Get_size() + 1)

        u_right = zeros(1)
        u_left = zeros(1)

        my_rank = COMM_WORLD.Get_rank()
        if my_rank > 0:
            COMM_WORLD.Send(u[:1], my_rank - 1)
        if my_rank < COMM_WORLD.Get_size() - 1:
            COMM_WORLD.Recv(u_right, my_rank + 1)

        if my_rank < COMM_WORLD.Get_size() - 1:
            COMM_WORLD.Send(u[-1:], my_rank + 1)
        if my_rank > 0:
            COMM_WORLD.Recv(u_left, my_rank - 1)

        u_ext = hstack([u_left, u, u_right])
        f = (u_ext[2:] + u_ext[:-2] - 2 * u_ext[1:-1]) / dx**2

        f_diff_u = diff_mpi(f, u, 'tangent')

        # check diagonal blocks
        lapl = -2 * sp.eye(N,N) \
             + sp.dia_matrix((np.ones(N), 1), (N,N)) \
             + sp.dia_matrix((np.ones(N), -1), (N,N))

        my_rank = COMM_WORLD.Get_rank()
        discrepancy = f_diff_u[my_rank] - lapl / dx**2
        if discrepancy.nnz > 0:
            self.assertAlmostEqual(0, np.abs(discrepancy.data).max())

        # lower diagonal blocks
        lapl_l = sp.csr_matrix(([1.], ([0], [N-1])), shape=(N,N))
        if my_rank > 0:
            discrepancy = f_diff_u[my_rank-1] - lapl_l / dx**2
            if discrepancy.nnz > 0:
                self.assertAlmostEqual(0, np.abs(discrepancy.data).max())

        # upper diagonal blocks
        lapl_u = lapl_l.T
        if my_rank < COMM_WORLD.Get_size() - 1:
            discrepancy = f_diff_u[my_rank+1] - lapl_u / dx**2
            if discrepancy.nnz > 0:
                self.assertAlmostEqual(0, np.abs(discrepancy.data).max())

        # other blocks are 0
        for rank in range(COMM_WORLD.Get_size()):
            if abs(rank - my_rank) > 1 and rank in f_diff_u:
                self.assertEqual(f_diff_u[rank], 0)


if __name__ == '__main__':
    from numpad import *

    # # 4th order derivative test
    # N = 4
    # u = random(N)
    # dx = 1. # / (N * COMM_WORLD.Get_size() + 1)

    # f = u
    # for i in range(2):
    #     f_right = zeros(1)
    #     f_left = zeros(1)

    #     my_rank = COMM_WORLD.Get_rank()
    #     if my_rank > 0:
    #         COMM_WORLD.Send(f[:1], my_rank - 1)
    #     if my_rank < COMM_WORLD.Get_size() - 1:
    #         COMM_WORLD.Recv(f_right, my_rank + 1)

    #     if my_rank < COMM_WORLD.Get_size() - 1:
    #         COMM_WORLD.Send(f[-1:], my_rank + 1)
    #     if my_rank > 0:
    #         COMM_WORLD.Recv(f_left, my_rank - 1)

    #     f_ext = hstack([f_left, f, f_right])
    #     f = (f_ext[2:] + f_ext[:-2] - 2 * f_ext[1:-1]) / dx**2

    # f_diff_u = diff_mpi(f, u, 'tangent')

    # for my_rank in range(COMM_WORLD.Get_size()):
    #     COMM_WORLD.Barrier()
    #     if my_rank != COMM_WORLD.Get_rank(): continue
    #     for rank in sorted(f_diff_u):
    #         if hasattr(f_diff_u[rank], 'todense'):
    #             print(my_rank, rank, f_diff_u[rank].todense())
    #         else:
    #             print(my_rank, rank, f_diff_u[rank])

    unittest.main()
