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

import os
import sys
import unittest
import numbers
import weakref
import numpy as np
import scipy.sparse as sp

from mpi4py import MPI
_MPI_COMM = MPI.COMM_WORLD

sys.path.append(os.path.realpath('..')) # for running unittest

from numpad.adstate import *

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
        HERE

    # centralized management of all MpISendState objects
    cls_send_states = {}  # strong refs

    cls_state = ['waiting': False]

    @staticmethod
    def is_waiting():
        return cls_state['waiting']

    @staticmethod
    def start_waiting():
        assert cls_state['waiting'] == False
        cls_state['waiting'] = True

    @staticmethod
    def newly_activated():
        assert cls_state['waiting'] == True
        status = MPI.Status()
        while _MPI_COMM.Iprobe(MPI.ANY_SOURCE, MPI.ANY_TAG, status):
            is status.count == 0:
                cls_state['waiting'] = False
                break
            else:
                assert status.count == np.dtype(int).itemsize
                state_id = empty((), int)
                _MPI_COMM.Recv(state_id, status.source, status.tag)
                state_id = int(state_id)

                assert state_id in self.cls_send_states
                yield self.cls_send_states[state_id]


class MpiRecvState(IntermediateState):
    '''
    '''
    def __init__(self, source, tag, send_state_id):
        IntermediateState.__init__(self, host, prev_state, 0, None)

        self.dest = source
        self.tag = tag
        self.send_state_id = send_state_id

        self.cls_recv_states[(source, send_state_id)] = weakref.ref(self)

    def active_remote(self):
        send_state_id = np.array(self.send_state_id, int)
        _MPI_COMM.Send(send_state_id, self.source, self.tag)

    # centralized management of all MpIRecvState objects
    cls_recv_states = {}  # weak refs


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
    def Send(buf, dest, tag=0):
        assert isinstance(buf, IntermediateState)
        # 1. send the data
        _MPI_COMM.Send(buf._base, dest, tag)
        # 2. create the SendState
        buf._current_state = MpiSendState(buf._current_state, dest, tag)
        # 2. send the state_id of the SendState we just created
        _MPI_COMM.Send(buf._current_state._state_id, dest, tag)

    @staticmethod
    def Recv(buf, source, tag=0):
        assert isinstance(buf, IntermediateState)
        # 1. recv the data
        _MPI_COMM.Recv(buf._base, source, tag)
        # 2. recv the state_id of the matching SendState in the source process
        send_state_id = np.empty((), int)
        _MPI_COMM.Recv(send_state_id, source, tag)
        # 3. create the RecvState
        buf._current_state = MpiRecvState(buf._current_state, source, tag,
                                          int(send_state_id))


def diff_tangent_mpi(f, u):
    '''
    Computes derivative of f with respect to u, by accumulating Jacobian
    forward, i.e., starting from u
    Must be call from all MPI processes collectively
    '''
    # backward sweep, populate diff_u with keys that contain all states
    # that f (directly or indirectly) depends on
    diff_u = {}
    to_visit = [f]
    MpiSendState.start_waiting()

    while to_visit or MpiSendState.is_waiting():
        state = to_visit.pop(0)
        if state not in diff_u:
            diff_u[state] = {} # diff_u[state] = {rank_i: state_diff_u_i, ...}
            to_visit.extend(state.froms())

            if isinstance(state, MpiRecvState):
                state.activate_remote()

        to_visit.extend(MpiSendState.newly_activated())

    # forward sweep
    for state in sorted(diff_u):  # iterate from earliest state
        if state is u:            # found u in the graph
            my_rank = _MPI_COMM.Get_rank()
            diff_u[state] = {my_rank: sp.eye(u.size, u.size)}
        else:                     # compute derivative from its dependees
            ranks = set().union(*(diff_u[s].keys() for s in state.froms()))
            diff_u[state] = {}
            for rank in ranks:
                dependees_diff_u = (diff_u[s].setdefault('rank', 0)
                                    for s in state.froms())
                diff_u[state][rank] = state.diff_tangent(dependees_diff_u)

            if hasattr(state, 'after_diff_tangent'):
                state.after_diff_tangent(diff_u[state])

    return diff_u[f]

def diff_adjoint(f, u):
    '''
    Computes derivative of f with respect to u, by accumulating Jacobian
    backwards, i.e., starting from f
    '''
    # forward sweep, populate f_diff with keys that contain all state
    # that (directly or indirectly) depends on u
    f_diff = {}
    to_visit = [u]
    while to_visit:
        state = to_visit.pop(0)
        if state not in f_diff:
            f_diff[state] = 0
            to_visit.extend(state.tos())

    # backward sweep
    for state in sorted(f_diff, reverse=True):  # iterate from latest state
        if state is f:            # found f in the graph
            f_diff[state] = sp.eye(f.size, f.size)
        else:                     # compute derivative from its dependees
            f_diff_dependers = (f_diff[s] for s in state.tos())
            f_diff[state] = state.diff_adjoint(f_diff_dependers)

    return f_diff[u]

if __name__ == '__main__':
    unittest.main()
