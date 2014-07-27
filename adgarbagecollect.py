# garbage collect states that can no longer be reached
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

import pdb
import gc
import unittest
from numpad.adarray import *
from numpad.adsolve import *

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

