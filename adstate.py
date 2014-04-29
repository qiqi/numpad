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
                other_state.to_list.append(weakref.ref(self))
                assert isinstance(other_state, IntermediateState)
                self.other = other_state

        self.to_list = []

    def next_state(self, multiplier, other_state=None, op_name=''):
        return IntermediateState(self.host(), self, multiplier, other_state,
                                 op_name)

    # --------- recursive functions for tangent differentiation -------- #

    def clear_self_diff_u(self):
        if hasattr(self, '_self_diff_u'):
            del self._self_diff_u
            if self.other:
                self.other.clear_self_diff_u()
            if self.prev:
                self.prev.clear_self_diff_u()
    
    def diff_recurse(self, u):
        if hasattr(self, '_self_diff_u'):
            return self._self_diff_u

        if u is self:             # found u in the graph
            self_diff_u = sp.eye(u.size, u.size)
        elif self.prev is None:   # initial state, dead end
            self_diff_u = 0
        elif self.other is None:  # unitary operation
            self_diff_u = _multiply_ops(self.multiplier,
                                        self.prev.diff_recurse(u))
        else:                     # binary operation
            self_diff_u_0 = self.prev.diff_recurse(u)
            self_diff_u_1 = _multiply_ops(self.multiplier,
                                          self.other.diff_recurse(u))
            self_diff_u = _add_ops(self_diff_u_0, self_diff_u_1)
    
        self._self_diff_u = self_diff_u
        return self_diff_u

    # --------- recursive functions for adjoint differentiation -------- #

    def clear_f_diff_self(self):
        if hasattr(self, '_f_diff_self'):
            del self._f_diff_self
            if hasattr(self, 'next') and self.next():
                self.next().clear_f_diff_self()
            for to_item in self.to_list:
                if to_item():
                    to_item().clear_f_diff_self()
    
    def adjoint_recurse(self, f):
        if hasattr(self, '_f_diff_self'):
            return self._f_diff_self

        if f is self:             # found f in the graph
            f_diff_self = sp.eye(f.size, f.size)
        else:
            f_diff_self = 0
            if hasattr(self, 'next') and self.next():
                if self.next().other:  # binary operation
                    next_diff_self = 1
                else:                  # unitary operation
                    next_diff_self = self.next().multiplier

                f_diff_next = self.next().adjoint_recurse(f)
                f_diff_self = _multiply_ops(f_diff_next, next_diff_self)

            for to_item in self.to_list:
                if to_item():
                    assert to_item().other is self
                    f_diff_item = to_item().adjoint_recurse(f)
                    item_diff_self = to_item().multiplier
                    f_diff_self_i = _multiply_ops(f_diff_item, item_diff_self)
                    f_diff_self = _add_ops(f_diff_self, f_diff_self_i)

        self._f_diff_self = f_diff_self
        return f_diff_self

# -------- Sparse Jacobian objects for delayed Jacobian construction ------- #

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
        if self._shape:
            return self._shape
        else:
            return (self.i.max() + 1, self.j.max() + 1)

    def tocsr(self):
        if not hasattr(self, '_mat'):
            self._mat = sp.csr_matrix((self.data, (self.i, self.j)),
                                      shape=self._shape)
            self._shape = self._mat.shape
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
