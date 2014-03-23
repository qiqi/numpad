import pdb
import gc
import unittest
from adarray import *
from adsolve import *

def collect(state):
    gc.collect()
    _collect_recurse(state)
    _clear_can_collect(state)

def _clear_can_collect(state):
    if hasattr(state, 'can_collect'):
        del state.can_collect
        if state.other:
            _clear_can_collect(state.other)
        if state.prev:
            _clear_can_collect(state.prev)
        if hasattr(state, 'residual') and state.residual:
            _clear_can_collect(state.residual)

def _collect_recurse(state):
    if hasattr(state, 'can_collect'):
        return state.can_collect

    state.can_collect = (state.host() is None)

    if state.other:
        if _collect_recurse(state.other):
            state.other = None
        else:
            state.can_collect = False

    if state.prev:
        if _collect_recurse(state.prev):
            state.prev = None
        else:
            state.can_collect = False

    if hasattr(state, 'residual') and state.residual:
        if _collect_recurse(state.residual):
            state.residual = None
        else:
            state.can_collect = False

    return state.can_collect

