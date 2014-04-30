import pdb
import os
import sys
import time
import unittest
import numbers
import weakref
import numpy as np
import scipy.sparse as sp

sys.path.append(os.path.realpath('..')) # for running unittest

# --------------- intermediate states and their operations ---------------- #

g_state_count = 0

def InitialState(host):
    return IntermediateState(host, None, None, None)

class IntermediateState:
    def __init__(self, host, prev_state, multiplier, other_state,
                 op_name=''):
        global g_state_count
        self._state_id = g_state_count
        g_state_count += 1

        self.host = weakref.ref(host)
        self.size = host.size

        self.op_name = op_name

        self.prev = prev_state
        if prev_state is not None:
            assert isinstance(prev_state, IntermediateState)
            prev_state.next = weakref.ref(self)

        if multiplier is None:       # initial state
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
        if hasattr(self, 'next') and self.next():
            yield self.next()
        for ref in self._to_refs:
            if ref():
                yield ref()

    def froms(self):
        if self.prev:
            yield self.prev
        if self.other:
            yield self.other

    def next_state(self, multiplier, other_state=None, op_name=''):
        return IntermediateState(self.host(), self, multiplier, other_state,
                                 op_name)

    # --------- functions for tangent and adjoint differentiation -------- #

    def diff_tangent(self, dependees_diff_u):
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
        f_diff_self = 0
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
    # backward sweep, populate diff_u with active states being keys
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
    # forward sweep, populate f_diff with active states being keys
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
    def __init__(self, data):
        self.data = data

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
        return tocsr(op0) * tocsr(op1)

if __name__ == '__main__':
    unittest.main()
