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

def InitialState(host):
    return IntermediateState(host, None, None, None)

class IntermediateState:
    def __init__(self, host, prev_state, multiplier, other_state):
        self.host = weakref.ref(host)
        self.size = host.size

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

    def next_state(self, multiplier, other_state=None):
        return IntermediateState(self.host(), self, multiplier, other_state)

    # --------------- recursive functions for differentiation --------------- #

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

        if u is self:             # found u in the tree
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

# -------------- auxiliary functions for differentiation ------------ #

def _multiply_ops(op0, op1):
    if op0 is 0 or op1 is 0:
        return 0
    else:
        return op0 * op1

def _add_ops(op0, op1):
    if op0 is 0:
        return op1
    elif op1 is 0:
        return op0
    else:
        return op0 + op1


if __name__ == '__main__':
    unittest.main()
