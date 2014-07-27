# adstate.py maintains intermediate states, and accumulates derivatives
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
import numbers
import weakref
import numpy as np
import scipy.sparse as sp

sys.path.append(os.path.realpath('..')) # for running unittest

# --------------- intermediate states and their operations ---------------- #

_g_state_count = 0

def InitialState(host):
    '''
    Returns an IntermediateState object with no dependees
    '''
    return IntermediateState(host, None, None, None)

class IntermediateState:
    '''
    A state that evolves from a previous state (prev_state)
    When a unitary operation creates this new IntermediateState,
        multiplier is the Jacobian of the new state with respect to
        the previous state.  other_state must be None.
    When a binary operation creates this new IntermedateState,
        multiplier is the Jacobian of the new state with respect to
        the other contributing state (other_state).  The Jacobian with
        respect to prev_state is identity.
    The string op_name is used only in visualization

    IntermediateStates can be hashed and compared. A newer state is "larger".
    '''
    def __init__(self, host, prev_state, multiplier, other_state,
                 op_name=''):
        global _g_state_count
        self._state_id = _g_state_count
        _g_state_count += 1

        self.host = weakref.ref(host)
        self.size = host.size

        self.op_name = op_name

        self.prev = prev_state
        if prev_state is not None:
            assert isinstance(prev_state, IntermediateState)
            prev_state.next = weakref.ref(self)

        if multiplier is None:       # initial state depends on nothing
            assert prev_state is None and other_state is None
            self.other = None
        else:
            self.multiplier = multiplier
            if other_state is None:  # unitary operation
                if not isinstance(multiplier, numbers.Number):  # not 1 or 0
                    assert multiplier.shape == (self.size, self.size)
                self.other = None
            else:                    # binary operation
                if not isinstance(multiplier, numbers.Number):  # not 1 or 0
                    assert multiplier.shape == (self.size, other_state.size)
                other_state._to_refs.append(weakref.ref(self))
                assert isinstance(other_state, IntermediateState)
                self.other = other_state

        self._to_refs = []

    def __hash__(self):
        return self._state_id

    def __lt__(self, other):
        return self._state_id < other._state_id

    def __eq__(self, other):
        return self._state_id == other._state_id

    def tos(self):
        '''
        Generate all states that immediately depends on this state
        '''
        if hasattr(self, 'next') and self.next():
            yield self.next()
        for ref in self._to_refs:
            if ref():
                yield ref()

    def froms(self):
        '''
        Generate all states that this state immediately depends on
        '''
        if self.prev:
            yield self.prev
        if self.other:
            yield self.other

    def obliviate(self):
        # remove self from the depender list (_to_refs) of its dependees
        for state in self.froms():
            state._to_refs = [ref for ref in state._to_refs \
                              if ref() is not self]
        # remove reference to dependees
        if self.prev:
            self.prev = None
        if self.other:
            self.other = None
        if hasattr(self, 'multiplier') and self.multiplier:
            del self.multiplier

    def next_state(self, multiplier, other_state=None, op_name=''):
        '''
        Generate a state that evolves from this state,
        either through a unitary operation (other_state is None)
                    or a binary operation.
        '''
        return IntermediateState(self.host(), self, multiplier, other_state,
                                 op_name)

    # --------- functions for tangent and adjoint differentiation -------- #

    def diff_tangent(self, dependees_diff_u):
        '''
        Given dependees_diff_u, the derivative of immediate dependees (froms())
        with respect to u, this function should return the derivative of
        this state with respect to the same u
        '''
        if self.prev is None:     # initial state, has 0 derivative to anything
            return 0
        elif self.other is None:  # unitary operation
            prev_diff_u, = dependees_diff_u
            return _multiply_ops(self.multiplier, prev_diff_u)
        else:                     # binary operation
            prev_diff_u, other_diff_u = dependees_diff_u
            return _add_ops(prev_diff_u, _multiply_ops(self.multiplier,
                                                       other_diff_u))

    def diff_adjoint(self, f_diff_dependers):
        '''
        Given dependers_diff_u, the derivative of f with respect to
        immediate dependers (tos()), this function should return the
        derivative of the same f with respect to this state.
        '''
        f_diff_self = 0
        # go over immediate dependers and the derivative of f w.r.t. them
        for state, f_diff_state in zip(self.tos(), f_diff_dependers):
            if state.other is self:                   # binary operation
                state_diff_self = state.multiplier
            elif state.prev is self and state.other:  # binary operation
                state_diff_self = 1
            else:                                     # unitary operation
                assert state.prev is self
                state_diff_self = state.multiplier

            f_diff_self = _add_ops(f_diff_self,
                    _multiply_ops(f_diff_state, state_diff_self))
        return f_diff_self

# -------- tangent and adjoint differentiation through state graph ------- #

def diff_tangent(f, u):
    '''
    Computes derivative of f with respect to u, by accumulating Jacobian
    forward, i.e., starting from u
    '''
    # backward sweep, populate diff_u with keys that contain all states
    # that f (directly or indirectly) depends on
    diff_u = {}
    to_visit = [f]
    while to_visit:
        state = to_visit.pop(0)
        if state not in diff_u:
            diff_u[state] = 0
            to_visit.extend(s for s in state.froms())

    # forward sweep
    for state in sorted(diff_u):  # iterate from earliest state
        if state is u:            # found u in the graph
            diff_u[state] = sp.eye(u.size, u.size)
        else:                     # compute derivative from its dependees
            dependees_diff_u = (diff_u[s] for s in state.froms())
            diff_u[state] = state.diff_tangent(dependees_diff_u)

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
            to_visit.extend(s for s in state.tos())

    # backward sweep
    for state in sorted(f_diff, reverse=True):  # iterate from latest state
        if state is f:            # found f in the graph
            f_diff[state] = sp.eye(f.size, f.size)
        else:                     # compute derivative from its dependees
            f_diff_dependers = (f_diff[s] for s in state.tos())
            f_diff[state] = state.diff_adjoint(f_diff_dependers)

    return f_diff[u]

# -------- Jacobian class construct sparse matrix only when needed ------- #

class dia_jac:
    '''
    A Jacobian represented by a diagonal matrix.
    '''
    def __init__(self, diag_entries):
        self.data = diag_entries

    @property
    def shape(self):
        return (self.data.size, self.data.size)

    def tocsr(self):
        if not hasattr(self, '_mat'):
            n = self.data.size
            indices = np.arange(n, dtype=int)
            indptr = np.arange(n+1, dtype=int)
            self._mat = sp.csr_matrix((self.data, indices, indptr))
        return self._mat

class csr_jac:
    '''
    Jacobian represented by a scipy.sparse.csr_matrix((data, (i, j))
    '''
    def __init__(self, data, i, j, shape=None):
        self.data = data
        self.i = i
        self.j = j
        self._shape = shape

    @property
    def shape(self):
        if self._shape is None:
            self._shape = (self.i.max() + 1, self.j.max() + 1)
        return self._shape

    def tocsr(self):
        if not hasattr(self, '_mat'):
            self._mat = sp.csr_matrix((self.data, (self.i, self.j)),
                                       shape=self._shape)
        return self._mat

def tocsr(jac):
    if isinstance(jac, (dia_jac, csr_jac)):
        return jac.tocsr()
    else:
        return jac

# ------------- addition and multiplication of sparse Jacobians ------------ #

def _add_ops(op0, op1):
    if op0 is 0:
        return op1
    elif op1 is 0:
        return op0
    else:
        return tocsr(op0) + tocsr(op1)

def _multiply_ops(op0, op1):
    if op0 is 0 or op1 is 0:
        return 0
    else:
        op = tocsr(op0) * tocsr(op1)
        if hasattr(op, 'nnz') and op.nnz == 0:
            op = 0
        return op

if __name__ == '__main__':
    unittest.main()
