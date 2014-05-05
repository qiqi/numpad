# visualize computational graph through graphviz (dot)
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

sys.path.append(os.path.realpath('..')) # for running unittest

from numpad.adarray import *
from numpad.adsolve import *

def _clear_dot_name(state):
    if hasattr(state, '_dot_name'):
        del state._dot_name
        if state.other:
            _clear_dot_name(state.other)
        if state.prev:
            _clear_dot_name(state.prev)
        if hasattr(state, 'residual') and state.residual:
            _clear_dot_name(state.residual)

def _dot_edge(fro, to, tag, other_attr=''):
    return '{0} -> {1} [label="{2}", {3}];\n'.format(fro, to, tag, other_attr)

def _dot_string(state):
    dot_string = ''
    if not hasattr(state, '_dot_name'):
        state._dot_name = 'S{0}_{1}'.format(state._state_id, state.size)
        if state.other:
            sub_string = _dot_string(state.other)
            dot_string = dot_string \
                       + _dot_edge(state._dot_name, state.other._dot_name,
                                   state.op_name) \
                       + sub_string
        if state.prev:
            sub_string = _dot_string(state.prev)
            dot_string = dot_string \
                       + _dot_edge(state._dot_name, state.prev._dot_name,
                                   state.op_name) \
                       + sub_string
        if hasattr(state, 'residual') and state.residual:
            sub_string = _dot_string(state.residual)
            dot_string = dot_string \
                       + _dot_edge(state._dot_name, state.residual._dot_name,
                                   state.op_name, 'color=red') \
                       + sub_string
    return dot_string

def dot(f):
    from numpad.adgarbagecollect import collect
    collect(f._current_state)
    dot_string = _dot_string(f._current_state)
    _clear_dot_name(f._current_state)
    return 'digraph G{\n' + dot_string + '}\n'

if __name__ == '__main__':
    # visualize 5 steps of Euler flow calculation

    def extend(w_interior):
        w = zeros([4, Ni+2, Nj+2])
        w[:,1:-1,1:-1] = w_interior.reshape([4, Ni, Nj])
        # inlet
        rho, u, v, E, p = primative(w[:,1,1:-1])
        c = sqrt(1.4 * p / rho)
        mach = u / c
        rhot = rho * (1 + 0.2 * mach**2)**2.5
        pt = p * (1 + 0.2 * mach**2)**3.5
    
        d_rho = 1 - rho
        d_pt = pt_in - pt
        d_u = d_pt / (rho * (u + c))
        d_p = rho * c * d_u
    
        relax = 0.5
        rho = rho + relax * d_rho
        u = u + relax * d_u
        p = p + relax * d_p
        w[0,0,1:-1] = rho
        w[1,0,1:-1] = rho * u
        w[2,0,1:-1] = 0
        w[3,0,1:-1] = p / 0.4 + 0.5 * rho * u**2
        # outlet
        w[:3,-1,1:-1] = w[:3,-2,1:-1]
        w[3,-1,1:-1] = p_out / (1.4 - 1) + \
                    0.5 * (w[1,-1,1:-1]**2 + w[2,-1,1:-1]**2) / w[0,-1,1:-1]
        # walls
        w[:,:,0] = w[:,:,1]
        w[2,:,0] *= -1
        w[:,:,-1] = w[:,:,-2]
        w[2,:,-1] *= -1
        return w
        
    def primative(w):
        rho = w[0]
        u = w[1] / rho
        v = w[2] / rho
        E = w[3]
        p = 0.4 * (E - 0.5 * (u * w[1] + v * w[2]))
        return rho, u, v, E, p
        
    def euler(w, w0, dt):
        import numpad
        w_ext = extend(w)
        rho, u, v, E, p = primative(w_ext)
        # cell center flux
        F = array([rho*u, rho*u**2 + p, rho*u*v, u*(E + p)])
        G = array([rho*u, rho*u*v, rho*v**2 + p, v*(E + p)])
        # interface flux
        Fx = 0.5 * (F[:,1:,1:-1] + F[:,:-1,1:-1])
        Fy = 0.5 * (F[:,1:-1,1:] + F[:,1:-1,:-1])
        # numerical viscosity
        C = 300
        Fx -= 0.5 * C * (w_ext[:,1:,1:-1] - w_ext[:,:-1,1:-1])
        Fy -= 0.5 * C * (w_ext[:,1:-1,1:] - w_ext[:,1:-1,:-1])
        C = 300
        Fx = -0.5 * C * (w_ext[:,1:,1:-1] - w_ext[:,:-1,1:-1])
        Fy = -0.5 * C * (w_ext[:,1:-1,1:] - w_ext[:,1:-1,:-1])
        # # residual
        divF = (Fx[:,1:,:] - Fx[:,:-1,:]) / dx + (Fy[:,:,1:] - Fy[:,:,:-1]) / dy
        return (w - w0) / dt + ravel(divF)

    # ---------------------- time integration --------------------- #
    Ni, Nj = 10, 5
    dx, dy = 10./ Ni, 1./Nj
    t, dt = 0, 1E-5
    
    pt_in = 1.2E5
    p_out = 1E5
    
    w0 = zeros([4, Ni, Nj])
    w0[0] = 1
    w0[3] = 1E5 / (1.4 - 1)
    
    w = ravel(w0)
    
    for i in range(2):
        print('t = ', t)
        w = solve(euler, w, args=(w, dt), rel_tol=1E-9, abs_tol=1E-7)
        t += dt

    open('graph.dot', 'wt').write(dot(w))
