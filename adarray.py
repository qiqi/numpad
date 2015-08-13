# adarray.py defines adarray that emulates numpy.ndarray
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
import os
import sys
import time
import unittest
import numbers
import weakref
import numpy as np
import scipy.sparse as sp

sys.path.append(os.path.realpath('..')) # for running unittest

from numpad.adstate import *

# ---------------- numpy equivalentsa ---------------- #
newaxis = np.newaxis

# ----------------- debug utilities ------------------ #

__DEBUG_MODE__ = False
__DEBUG_TOL__ = None
__DEBUG_SEED_ARRAYS__ = []

def _DEBUG_perturb_enable(enable=True, tolerance=None):
    '''
    Turn __DEBUG_MODE__ on.
    If you call this function, you should call it first thing after importing
    numpad.
    All indepedent variables generate random perturbations, and all dependent
    variables propagate these perturbations.  All arithmetic operations in
    which Automatic Differentiation is performed are verified against these
    perturbations.  If tolerance is set, AssertionError is raised when the
    AD derivative and the perturbations differ by more than the tolerance.
    '''
    global __DEBUG_MODE__, __DEBUG_TOL__
    assert(isinstance(enable, bool))
    __DEBUG_MODE__ = enable
    __DEBUG_TOL__ = tolerance

def _DEBUG_perturb_verify(output, message=''):
    '''
    If __DEBUG_MODE__ is on, verify a dependent variable "output" against
    random perturbations, print the error norm of the discrepancy,
    generate AssertionError if the error norm exceeds the tolerance.
    '''
    if not __DEBUG_MODE__: return
    assert np.isfinite(output._value).all()
    out_perturb = np.zeros(output.size)
    for var, var_perturb in __DEBUG_SEED_ARRAYS__:
        J = output.diff(var)
        if J is not 0:
            out_perturb += J * var_perturb
    error_norm = np.linalg.norm(out_perturb - np.ravel(output._DEBUG_perturb))
    print('_DEBUG_perturb_verify ', message, ': ', error_norm)
    if __DEBUG_TOL__:
        assert error_norm < __DEBUG_TOL__

def _DEBUG_perturb_new(var):
    '''
    Generate a random perturbation for a given independent variable "var".
    Return "var", which is now associated with the new random perturbation.
    '''
    global __DEBUG_MODE__, __DEBUG_SEED_ARRAYS__
    if __DEBUG_MODE__:
        var._DEBUG_perturb = np.random.random(var.shape)
        __DEBUG_SEED_ARRAYS__.append((var, np.ravel(var._DEBUG_perturb.copy())))
    return var

def _DEBUG_perturb_retrieve(var):
    '''
    Retrieve the random perturbation associated with variable "var".
    '''
    if hasattr(var, '_DEBUG_perturb'):
        return var._DEBUG_perturb
    elif isinstance(var, adarray):
        return np.zeros(var.shape)
    else:
        return np.zeros(np.asarray(var).shape)

def adarray_count():
    import gc
    gc.collect()
    return len([obj for obj in gc.get_objects() if isinstance(obj, adarray)])

def adstate_count():
    import gc
    gc.collect()
    return len([obj for obj in gc.get_objects() \
                if isinstance(obj, IntermediateState)])

# --------------------- utilities --------------------- #

def value(a):
    '''
    Return the "value" of an adarray "a".  The value is a numpy.ndarray
    object containing all the data of a.
    If a is a number of a numpy.ndarray, then return a itself.
    '''
    if isinstance(a, (numbers.Number, np.ndarray, list)):
        return a
    else:
        return a._value

# --------------------- adarray construction --------------------- #
def append_docstring_from_numpy(f):
    '''
    Decorator for appending numpy docstring to numpad function docstring
    '''
    def f_more_doc(*args, **kargs):
        return f(*args, **kargs)

    try:
        import numpy
        f_name = f.__qualname__.split('.')[-1]
        numpy_doc = eval('numpy.{0}'.format(f_name)).__doc__
    except:
        numpy_doc = ''

    if not f.__doc__:
        f.__doc__ = '\nOverloaded by numpad, returns adarray.\n'
    f_more_doc.__doc__ = f.__doc__ + numpy_doc
    return f_more_doc


@append_docstring_from_numpy
def zeros(*args, **kargs):
    return array(np.zeros(*args, **kargs))

@append_docstring_from_numpy
def ones(*args, **kargs):
    return array(np.ones(*args, **kargs))

@append_docstring_from_numpy
def empty(*args, **kargs):
    return array(np.empty(*args, **kargs))

@append_docstring_from_numpy
def eye(*args, **kargs):
    return array(np.eye(*args, **kargs))

@append_docstring_from_numpy
def linspace(*args, **kargs):
    return array(np.linspace(*args, **kargs))

@append_docstring_from_numpy
def load(*args, **kargs):
    return array(np.load(*args, **kargs))

@append_docstring_from_numpy
def loadtxt(*args, **kargs):
    return array(np.loadtxt(*args, **kargs))

# --------------------- algebraic functions --------------------- #

# def maximum(a, b):
#     a_gt_b = a > b
#     return a * a_gt_b + b * (1. - a_gt_b)
# 
# def minimum(a, b):
#     a_gt_b = a > b
#     return b * a_gt_b + a * (1. - a_gt_b)

def sigmoid(x):
    return (tanh(x) + 1) / 2

def gt_smooth(a, b, c=0.1):
    return sigmoid((a - b) / c)

def lt_smooth(a, b, c=0.1):
    return sigmoid((b - a) / c)

def maximum_smooth(a, b, c=0.1):
    a_gt_b = gt_smooth(a, b, c)
    return a * a_gt_b + b * (1. - a_gt_b)

def minimum_smooth(a, b, c=0.1):
    return -maximum_smooth(-a, -b, c)

@append_docstring_from_numpy
def exp(x, out=None):
    x = array(x)

    if out is None:
        out = adarray(np.exp(x._value))
    else:
        np.exp(x._value, out._value)
        out.next_state(0, op_name='0')

    multiplier = dia_jac(np.exp(np.ravel(x._value)))
    out.next_state(multiplier, x, op_name='exp')

    if __DEBUG_MODE__:
        out._DEBUG_perturb = np.exp(x._value) * _DEBUG_perturb_retrieve(x)
        _DEBUG_perturb_verify(out)
    return out

@append_docstring_from_numpy
def sqrt(x):
    return x**(0.5)

@append_docstring_from_numpy
def sin(x, out=None):
    x = array(x)

    if out is None:
        out = adarray(np.sin(x._value))
    else:
        np.sin(x._value, out._value)
        out.next_state(0, op_name='0')
    multiplier = dia_jac(np.cos(np.ravel(x._value)))
    out.next_state(multiplier, x, op_name='sin')

    if __DEBUG_MODE__:
        out._DEBUG_perturb = np.cos(x._value) * _DEBUG_perturb_retrieve(x)
        _DEBUG_perturb_verify(out)
    return out

@append_docstring_from_numpy
def cos(x, out=None):
    x = array(x)

    if out is None:
        out = adarray(np.cos(x._value))
    else:
        np.cos(x._value, out._value)
        out.next_state(0, op_name='0')
    multiplier = dia_jac(-np.sin(np.ravel(x._value)))
    out.next_state(multiplier, x, 'cos')

    if __DEBUG_MODE__:
        out._DEBUG_perturb = -np.sin(x._value) * _DEBUG_perturb_retrieve(x)
        _DEBUG_perturb_verify(out)
    return out

@append_docstring_from_numpy
def log(x, out=None):
    x = array(x)

    if out is None:
        out = adarray(np.log(x._value))
    else:
        np.log(x._value, out._value)
        out.next_state(0, op_name='0')
    multiplier = dia_jac(1. / np.ravel(x._value))
    out.next_state(multiplier, x, 'log')

    if __DEBUG_MODE__:
        out._DEBUG_perturb = _DEBUG_perturb_retrieve(x) / x._value
        _DEBUG_perturb_verify(out)
    return out

@append_docstring_from_numpy
def tanh(x, out=None):
    x = array(x)

    if out is None:
        out = adarray(np.tanh(x._value))
    else:
        np.tanh(x._value, out._value)
        out.next_state(0, op_name='0')

    multiplier = dia_jac(1 - np.tanh(np.ravel(x._value))**2)
    out.next_state(multiplier, x, 'tanh')

    if __DEBUG_MODE__:
        out._DEBUG_perturb = _DEBUG_perturb_retrieve(x) \
                           * (1 - np.tanh(x._value)**2)
        _DEBUG_perturb_verify(out)
    return out

# ------------------ copy, stack, transpose operations ------------------- #

@append_docstring_from_numpy
def array(a):
    if isinstance(a, adarray):
        return a
    elif isinstance(a, (numbers.Number, np.ndarray)):
        a = adarray(a)
        _DEBUG_perturb_new(a)
        return a
    elif isinstance(a, (list, tuple)):
        a = list(a)
        # recursively convert subcomponents into adarrays
        for i in range(len(a)):
            a[i] = array(a[i])
        # make big array and add multipliers
        adarray_a = adarray(np.array([ai._value for ai in a]))
        for i in range(len(a)):
            data = np.ones(a[i].size)
            j_data = np.arange(a[i].size)
            i_data = i * a[i].size + j_data
            shape = (adarray_a.size, a[i].size)
            multiplier = csr_jac(data, i_data, j_data, shape=shape)
            adarray_a.next_state(multiplier, a[i], 'array')

        if __DEBUG_MODE__:
            _DEBUG_perturb_list = []
            for i in range(len(a)):
                _DEBUG_perturb_list.append(_DEBUG_perturb_retrieve(a[i]))
            adarray_a._DEBUG_perturb = np.array(_DEBUG_perturb_list)
            _DEBUG_perturb_verify(adarray_a)

        return adarray_a
        
@append_docstring_from_numpy
def ravel(a):
    a = array(a)
    return a.reshape((a.size,))

@append_docstring_from_numpy
def copy(a):
    a_copy = adarray(np.copy(value(a)))
    if isinstance(a, adarray):
        a_copy.next_state(1, a, 'cpy')
        if __DEBUG_MODE__:
            a_copy._DEBUG_perturb = _DEBUG_perturb_retrieve(a).copy()
            _DEBUG_perturb_verify(a_copy)
    else:
        assert isinstance(a, np.ndarray)
    return a_copy

@append_docstring_from_numpy
def transpose(a, axes=None):
    a = array(a)
    a_transpose = adarray(np.transpose(a._value, axes))
    i = np.arange(a.size).reshape(a.shape)
    j = np.transpose(i, axes)
    data = np.ones(i.size)
    multiplier = csr_jac(data, np.ravel(i), np.ravel(j))
    a_transpose.next_state(multiplier, a, 'T')
    if __DEBUG_MODE__:
        a_transpose._DEBUG_perturb = _DEBUG_perturb_retrieve(a).T
        _DEBUG_perturb_verify(a_transpose)
    return a_transpose

@append_docstring_from_numpy
def concatenate(adarrays, axis=0):
    adarrays = [array(a) for a in adarrays]
    ndarrays, marker_arrays = [], []
    for a in adarrays:
        if a.ndim == 0:
            a = a[np.newaxis]
        marker_arrays.append(len(ndarrays) * np.ones_like(a._value))
        ndarrays.append(value(a))

    concatenated_array = adarray(np.concatenate(ndarrays, axis))
    marker = np.ravel(np.concatenate(marker_arrays, axis))

    # marker now contains integers corresponding to which component
    for i_component, a in enumerate(adarrays):
        i = (marker == i_component).nonzero()[0]
        j = np.arange(i.size)
        data = np.ones(i.size, int)
        multiplier = csr_jac(data, i, j, shape=(marker.size, i.size))
        concatenated_array.next_state(multiplier, a, 'cat')

    if __DEBUG_MODE__:
        _DEBUG_perturb_list = []
        for a in adarrays:
            _DEBUG_perturb_list.append(_DEBUG_perturb_retrieve(a))
        concatenated_array._DEBUG_perturb = \
            np.concatenate(_DEBUG_perturb_list, axis)
        _DEBUG_perturb_verify(concatenated_array)
    return concatenated_array

@append_docstring_from_numpy
def hstack(adarrays):
    max_ndim = max(array(a).ndim for a in adarrays)
    axis = 1 if max_ndim > 1 else 0
    return concatenate(adarrays, axis=axis)

@append_docstring_from_numpy
def vstack(adarrays):
    max_ndim = max(array(a).ndim for a in adarrays)
    if max_ndim <= 1:
        return array(adarrays)
    else:
        return concatenate(adarrays, axis=0)

@append_docstring_from_numpy
def meshgrid(x, y):
    ind_xx, ind_yy = np.meshgrid(x._ind, y._ind)
    return x[ind_xx], y[ind_yy]

@append_docstring_from_numpy
def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    assert dtype is None and out is None
    a = array(a)
    sum_a = adarray(np.sum(a._value, axis, keepdims=keepdims))

    shape = np.sum(a._value, axis, keepdims=True).shape
    j = np.arange(sum_a.size).reshape(shape)
    i = np.ravel(j + np.zeros_like(a._value, int))
    j = np.ravel(a._ind)
    data = np.ones(i.size, int)
    multiplier = csr_jac(data, i, j, shape=(sum_a.size, a.size))
    sum_a.next_state(multiplier, a, 'sum')

    if __DEBUG_MODE__:
        sum_a._DEBUG_perturb = np.sum(a._DEBUG_perturb, axis, keepdims=keepdims)
        _DEBUG_perturb_verify(sum_a)
    return sum_a

@append_docstring_from_numpy
def mean(a, axis=None, dtype=None, out=None, keepdims=False):
    sum_a = sum(a, axis, dtype, out, keepdims)
    return sum_a * (float(sum_a.size) / a.size)

@append_docstring_from_numpy
def rollaxis(a, axis, start=0):
    b = adarray(np.rollaxis(a._value, axis, start))

    data = np.ones(a.size)
    j = np.ravel(a._ind)
    i = np.ravel(np.rollaxis(a._ind, axis, start))
    multiplier = csr_jac(data, i, j)

    b.next_state(multiplier, a, 'rollaxis')
    return b

@append_docstring_from_numpy
def roll(a, shift, axis=None):
    b = adarray(np.roll(a._value, shift, axis))

    data = np.ones(a.size)
    i = np.ravel(a._ind)
    j = np.ravel(np.roll(a._ind, shift, axis))
    multiplier = csr_jac(data, i, j)

    b.next_state(multiplier, a, 'roll')
    return b

@append_docstring_from_numpy
def dot(a, b):
    dot_axis = a.ndim - 1  # axis to sum over
    if b.ndim > 1:
        # extend the dimension of a
        a = a.reshape(a.shape + ((1,) * (b.ndim - 1)))
        # roll axes of b so that the second last index is the first
        b = rollaxis(b, -2)
    # extend the dimension of b
    b = b.reshape(((1,) * (a.ndim - b.ndim)) + b.shape)
    return sum(a * b, axis=dot_axis)

# ===================== the adarray class ====================== #

class adarray:
    dtype = np.dtype('float64')

    def __init__(self, array):
        self._value = np.asarray(value(array), np.float64)
        self._ind = np.arange(self.size).reshape(self.shape)
        self._current_state = InitialState(self)

    def __array__(self):
        return self._value

    def _ind_casted_to(self, shape):
        ind = np.zeros(shape, dtype=int)
        if ind.ndim:
            ind[:] = self._ind
        else:
            ind = self._ind
        return ind

    @property
    def _initial_state(self):
        state = self._current_state
        while state.prev:
            state = state.prev
        return state

    def next_state(self, multiplier, other=None, op_name=''):
        if other is None:
            self._current_state = \
                    self._current_state.next_state(multiplier, None, op_name)
        elif isinstance(other, adarray):
            self._current_state = \
                    self._current_state.next_state(multiplier,
                                other._current_state, op_name)
        else:
            raise NotImplementedError

    @property
    def size(self):
        return self._value.size
    @property
    def shape(self):
        return self._value.shape
    @property
    def ndim(self):
        return self._value.ndim
    @property
    def T(self):
        return self.transpose()

    def __len__(self):
        return self._value.__len__()

    def obliviate(self):
        self._current_state.obliviate()

    # ------------------ object operations ----------------- #

    def copy(self):
        return copy(self)

    def transpose(self, axes=None):
        return transpose(self, axes)

    def reshape(self, shape):
        reshaped = adarray(self._value.reshape(shape))
        if self.size > 0:
            reshaped.next_state(1, self, 'reshape')
        if __DEBUG_MODE__:
            reshaped._DEBUG_perturb = self._DEBUG_perturb.reshape(shape)
        return reshaped

    def sort(self, axis=-1, kind='quicksort'):
        '''
        sort in place
        '''
        ind = np.argsort(self._value, axis, kind)
        self._value.sort(axis, kind)

        j = np.ravel(self._ind[ind])
        i = np.arange(j.size)
        multiplier = csr_jac(np.ones(j.size), i,j, shape=(j.size, self.size))
        self.next_state(multiplier, op_name='sort')

        if __DEBUG_MODE__:
            self._DEBUG_perturb = _DEBUG_perturb_retrieve(self)[ind]
            _DEBUG_perturb_verify(self)

    # ------------------ boolean operations ----------------- #

#     def __eq__(self, a):
#         return array(self._value == value(a))
# 
#     def __ne__(self, a):
#         return array(self._value != value(a))
# 
#     def __gt__(self, a):
#         return array(self._value > value(a))
#     
#     def __ge__(self, a):
#         return array(self._value >= value(a))
# 
#     def __lt__(self, a):
#         return array(self._value < value(a))
#     
#     def __le__(self, a):
#         return array(self._value <= value(a))
# 
#     def all(self):
#         return self._value.all()

    # ------------------ arithmetic operations ----------------- #

    def __add__(self, a):
        if isinstance(a, numbers.Number):
            a_p_b = adarray(self._value + a)
            a_p_b.next_state(1, self, '+')
        else:
            b = self
            a_p_b = adarray(value(a) + value(b))

            if a.shape == b.shape:
                if hasattr(a, '_value'):
                    a_p_b.next_state(1, a, '+')
                if hasattr(b, '_value'):
                    a_p_b.next_state(1, b, '+')
            else:
                # a, b, or both is "broadcasted" to fit the shape of each other
                multiplier = np.ones(a_p_b.shape)
                i = np.arange(a_p_b.size)

                if hasattr(a, '_value') and multiplier.size > 0:
                    j_a = np.ravel(a._ind_casted_to(a_p_b.shape))
                    a_multiplier = csr_jac(np.ravel(multiplier), i, j_a,
                                           shape=(a_p_b.size, a.size))
                    a_p_b.next_state(a_multiplier, a, '+')
                if hasattr(b, '_value') and multiplier.size > 0:
                    j_b = np.ravel(b._ind_casted_to(a_p_b.shape))
                    b_multiplier = csr_jac(np.ravel(multiplier), i, j_b,
                                           shape=(a_p_b.size, b.size))
                    a_p_b.next_state(b_multiplier, b, '+')

        if __DEBUG_MODE__:
            a_p_b._DEBUG_perturb = _DEBUG_perturb_retrieve(self) \
                                 + _DEBUG_perturb_retrieve(a)
            _DEBUG_verify(a_p_b)
        return a_p_b

    def __radd__(self, a):
        return self.__add__(a)

    def __iadd__(self, a):
        if isinstance(a, (numbers.Number, np.ndarray)):
            self._value += a
        else:
            self._value += a._value
            if a.shape == self.shape:
                self.next_state(1, a, '+')
            else:
                # a is broadcasted to fit self's shape
                raise NotImplementedError

        if __DEBUG_MODE__:
            self._DEBUG_perturb += _DEBUG_perturb_retrieve(a)
            _DEBUG_perturb_verify(self)
        return self

    def __neg__(self):
        neg_self = adarray(-self._value)
        neg_self.next_state(-1, self, '-')
        if __DEBUG_MODE__:
            neg_self._DEBUG_perturb = -_DEBUG_perturb_retrieve(self)
            _DEBUG_perturb_verify(neg_self)
        return neg_self

    def __pos__(self):
        return self
        
    def __sub__(self, a):
        return self.__add__(-a)

    def __rsub__(self, a):
        return (-self) + a

    def __isub__(self, a):
        return self.__iadd__(-a)

    def __mul__(self, a):
        if isinstance(a, numbers.Number):
            a_x_b = adarray(self._value * a)
            a_x_b.next_state(a, self, '*')
            if __DEBUG_MODE__:
                a_x_b._DEBUG_perturb = _DEBUG_perturb_retrieve(self) * a
                _DEBUG_perturb_verify(a_x_b)
        else:
            b = self
            a_x_b = adarray(value(a) * value(b))

            if a.shape == b.shape:
                if hasattr(a, '_value'):
                    a_x_b.next_state(dia_jac(value(b).ravel()), a, '*')
                if hasattr(b, '_value'):
                    a_x_b.next_state(dia_jac(value(a).ravel()), b, '*')
            else:
                a_multiplier = np.zeros(a_x_b.shape)
                b_multiplier = np.zeros(a_x_b.shape)
                if a_multiplier.ndim:
                    a_multiplier[:] = value(b)
                else:
                    a_multiplier = value(b).copy()
                if b_multiplier.ndim:
                    b_multiplier[:] = value(a).copy()
                else:
                    b_multiplier = value(a).copy()

                i = np.arange(a_x_b.size)

                if hasattr(a, '_value') and a_x_b.size > 0:
                    j_a = np.ravel(a._ind_casted_to(a_x_b.shape))
                    a_multiplier = csr_jac(np.ravel(a_multiplier), i, j_a,
                                           shape=(a_x_b.size, a.size))
                    a_x_b.next_state(a_multiplier, a, '*')
                if hasattr(b, '_value') and a_x_b.size > 0:
                    j_b = np.ravel(b._ind_casted_to(a_x_b.shape))
                    b_multiplier = csr_jac(np.ravel(b_multiplier), i, j_b,
                                           shape=(a_x_b.size, b.size))
                    a_x_b.next_state(b_multiplier, b, '*')

            if __DEBUG_MODE__:
                a_x_b._DEBUG_perturb = _DEBUG_perturb_retrieve(self) * value(a) \
                                     + value(self) * _DEBUG_perturb_retrieve(a)
                _DEBUG_perturb_verify(a_x_b)
        return a_x_b

    def __rmul__(self, a):
        return self.__mul__(a)

    def __imul__(self, a):
        if isinstance(a, numbers.Number):
            self._value *= a
            self.next_state(a, op_name='*')
            if __DEBUG_MODE__:
                self._DEBUG_perturb *= a
                _DEBUG_perturb_verify(self)
        else:
            multiplier = dia_jac(np.ravel(a._value.copy()))
            self.next_state(multiplier, op_name='*')
            multiplier = dia_jac(np.ravel(self._value.copy()))
            self.next_state(multiplier, a, '*')
            self._value *= a._value
            if __DEBUG_MODE__:
                self._DEBUG_perturb = _DEBUG_perturb_retrieve(self) * value(a) \
                                    + value(self) * _DEBUG_perturb_retrieve(a)
                _DEBUG_perturb_verify(self)
        return self

    def __div__(self, a):
        return self * a**(-1)

    def __rdiv__(self, a):
        return a * self**(-1)

    def __truediv__(self, a):
        return self * a**(-1)

    def __rtruediv__(self, a):
        return a * self**(-1)

    def __pow__(self, a):
        if not isinstance(a, numbers.Number):
            return NotImplemented
        self_to_a = adarray(self._value ** a)
        multiplier = a * np.ravel(self._value)**(a-1)
        if multiplier.size > 0:
            multiplier[~np.isfinite(multiplier)] = 0
            multiplier = dia_jac(multiplier)
            self_to_a.next_state(multiplier, self, '**')
        if __DEBUG_MODE__:
            self_to_a._DEBUG_perturb = a * self._value**(a-1) \
                                     * _DEBUG_perturb_retrieve(self)
            _DEBUG_perturb_verify(self_to_a)
        return self_to_a

    def __rpow__(self, a):
        return exp(self * log(a))
    
    def sum(self, axis=None, dtype=None, out=None):
        return sum(self, axis, dtype=None, out=None)

    def mean(self, axis=None, dtype=None, out=None):
        return mean(self, axis, dtype=None, out=None)

    # ------------------ indexing ----------------- #

    def __getitem__(self, ind):
        self_i = adarray(self._value[ind])

        j = np.ravel(self._ind[ind])
        if j.size > 0:
            i = np.arange(j.size)
            multiplier = csr_jac(np.ones(j.size), i,j,
                                 shape=(j.size, self.size))
            self_i.next_state(multiplier, self, '[]')

        if __DEBUG_MODE__:
            self_i._DEBUG_perturb = _DEBUG_perturb_retrieve(self)[ind]
            _DEBUG_perturb_verify(self_i)
        return self_i

    def __setitem__(self, ind, a):
        a = array(a)
        data = np.ones(self.size)
        data[np.ravel(self._ind[ind])] = 0
        multiplier = dia_jac(data)
        self.next_state(multiplier, op_name='[]=0')

        self._value.__setitem__(ind, value(a))

        if hasattr(a, '_value'):
            i = self._ind[ind]
            if i.size > 0:
                j = a._ind_casted_to(i.shape)
                i, j = np.ravel(i), np.ravel(j)
                multiplier = csr_jac(np.ones(j.size), i,j,
                                     shape=(self.size, a.size))
                self.next_state(multiplier, a, op_name='[]')

        if __DEBUG_MODE__:
            self._DEBUG_perturb[ind] = _DEBUG_perturb_retrieve(a)
            _DEBUG_perturb_verify(self)

    # ------------------ str, repr ------------------ #

    def __str__(self):
        return str(self._value)

    def __repr__(self):
        return 'ad' + repr(self._value)

    def diff(self, u, mode='auto'):
        return diff(self, u, mode)

# ------------- replace numpy operations ------------- #

if np.set_numeric_ops()['add'] == np.add:
    def _add(x1, x2, out=None):
        if isinstance(x2, adarray):
            return x2.__add__(x1)
        else:
            return np.add(x1, x2, out)
    np.set_numeric_ops(add=_add)

if np.set_numeric_ops()['subtract'] == np.subtract:
    def _sub(x1, x2, out=None):
        if isinstance(x2, adarray):
            return (-x2).__add__(x1)
        else:
            return np.subtract(x1, x2, out)
    np.set_numeric_ops(subtract=_sub)

if np.set_numeric_ops()['multiply'] == np.multiply:
    def _mul(x1, x2, out=None):
        if isinstance(x2, adarray):
            return x2.__mul__(x1)
        else:
            return np.multiply(x1, x2, out)
    np.set_numeric_ops(multiply=_mul)

if np.set_numeric_ops()['true_divide'] == np.true_divide:
    def _div(x1, x2, out=None):
        if isinstance(x2, adarray):
            return (x2**(-1)).__mul__(x1)
        else:
            return np.true_divide(x1, x2, out)
    np.set_numeric_ops(divide=_div)
    np.set_numeric_ops(true_divide=_div)

# ------------------ differentiation ------------------ #

def diff(f, u, mode='auto'):
    if mode == 'auto':
        if u.size < f.size:
            mode = 'tangent'
        else:
            mode = 'adjoint'

    if mode == 'tangent':
        derivative = diff_tangent(f._current_state, u._initial_state)
    elif mode == 'adjoint':
        derivative = diff_adjoint(f._current_state, u._initial_state)
    else:
        raise NotImplementedError

    if isinstance(derivative, numbers.Number) and derivative == 0:
        derivative = sp.csr_matrix((f.size, u.size), dtype=float)
    return derivative


class replace__globals__:
    def __init__(self, f):
        self.f = f
        self.old_globals = {}
        self.new_globals = {}
        import numpad
        for key in f.__globals__:
            if f.__globals__[key] is np:
                self.old_globals[key] = f.__globals__[key]
                self.new_globals[key] = numpad
            elif hasattr(f.__globals__[key], '__module__') \
            and str(f.__globals__[key].__module__).startswith('numpy') \
            and key in numpad.__dict__:
                self.old_globals[key] = f.__globals__[key]
                self.new_globals[key] = numpad.__dict__[key]

    def __call__(self, *args, **argv):
        for key, val in self.new_globals.items():
            self.f.__globals__[key] = val
        result = self.f(*args, **argv)
        for key, val in self.old_globals.items():
            self.f.__globals__[key] = val
        return result


def diff_func(func, u, args=(), kargs={}):
    u = adarray(value(u).copy())
    func = replace__globals__(func)
    fu = func(u, *args, **kargs)
    return fu.diff(u)

# =========================================================== #
#                                                             #
#                         unittests                           #
#                                                             #
# =========================================================== #

class _NumpyCastTest(unittest.TestCase):
    def testAddSubCast(self):
        N = 10
        a = np.ones(N)
        b = ones(N)
        self.assertTrue(isinstance(a + b, adarray))
        self.assertTrue(isinstance(b + a, adarray))
        self.assertTrue(isinstance(b + b, adarray))
        self.assertTrue(isinstance(a + a, np.ndarray))
        self.assertTrue(isinstance(a - b, adarray))
        self.assertTrue(isinstance(b - a, adarray))
        self.assertTrue(isinstance(b - b, adarray))
        self.assertTrue(isinstance(a - a, np.ndarray))

    def testMulDivCast(self):
        N = 10
        a = np.ones(N)
        b = ones(N)
        self.assertTrue(isinstance(a * b, adarray))
        self.assertTrue(isinstance(b * a, adarray))
        self.assertTrue(isinstance(b * b, adarray))
        self.assertTrue(isinstance(a * a, np.ndarray))
        self.assertTrue(isinstance(a / b, adarray))
        self.assertTrue(isinstance(b / a, adarray))
        self.assertTrue(isinstance(b / b, adarray))
        self.assertTrue(isinstance(a / a, np.ndarray))

class _ManipulationTest(unittest.TestCase):
    def testArray(self):
        print('testArray')
        N = 10
        a = random.random(N)
        b = random.random(N)
        c = random.random(N)
        d = random.random(N)
        e = array([[a, b], [c, d]])

        def analytical(pos):
            ind = np.r_[:N]
            ind_ptr = np.zeros(N * pos), np.r_[:N+1], N * np.ones(N * (3-pos))
            ind_ptr = np.hstack(ind_ptr)
            shape = 4 * N, N
            return sp.csr_matrix((np.ones(N), ind, ind_ptr), shape=shape)
        self.assertEqual(0, (e.diff(a, 'tangent') - analytical(0)).nnz)
        self.assertEqual(0, (e.diff(a, 'adjoint') - analytical(0)).nnz)
        self.assertEqual(0, (e.diff(b, 'tangent') - analytical(1)).nnz)
        self.assertEqual(0, (e.diff(b, 'adjoint') - analytical(1)).nnz)
        self.assertEqual(0, (e.diff(c, 'tangent') - analytical(2)).nnz)
        self.assertEqual(0, (e.diff(c, 'adjoint') - analytical(2)).nnz)
        self.assertEqual(0, (e.diff(d, 'tangent') - analytical(3)).nnz)
        self.assertEqual(0, (e.diff(d, 'adjoint') - analytical(3)).nnz)

    def testDot(self):
        print('testDot')
        N = 20
        a = random.random((10, N))
        b = random.random((N, 30))
        c = dot(a, b)

        c_diff_b = c.diff(b)
        discrepancy = c_diff_b - sp.kron(a._value, sp.eye(c.shape[1]))
        if discrepancy.nnz > 0:
            self.assertAlmostEqual(0, np.abs(discrepancy.data).max())

        c_diff_a = c.diff(a)
        discrepancy = c_diff_a - sp.kron(sp.eye(c.shape[0]), b.T._value)
        if discrepancy.nnz > 0:
            self.assertAlmostEqual(0, np.abs(discrepancy.data).max())

    def testRoll(self):
        print('testRoll')
        N = 10
        a = linspace(1, N, N)
        b = roll(a, -1)
        dbda = b.diff(a)
        self.assertAlmostEqual(0, value((b[:-1] - a[1:])**2).sum())
        self.assertAlmostEqual(0, dbda[0,0])
        self.assertAlmostEqual(1, dbda[0,1])

    def testTranspose(self):
        print('testTranspose')
        N = 10
        a = random.random(N)
        b = random.random(N)
        c = transpose([a, b])

        i, j = np.arange(N) * 2, np.arange(N)
        c_diff_a = sp.csr_matrix((np.ones(N), (i,j)), shape=(2*N, N))
        i, j = np.arange(N) * 2 + 1, np.arange(N)
        c_diff_b = sp.csr_matrix((np.ones(N), (i,j)), shape=(2*N, N))
        self.assertEqual(0, (c.diff(a, 'tangent') - c_diff_a).nnz)
        self.assertEqual(0, (c.diff(a, 'adjoint') - c_diff_a).nnz)
        self.assertEqual(0, (c.diff(b, 'tangent') - c_diff_b).nnz)
        self.assertEqual(0, (c.diff(b, 'adjoint') - c_diff_b).nnz)


class _IndexingTest(unittest.TestCase):
    def test1DIndexByInt(self):
        print('testIndex')
        N = 10
        i = [2,5,-1]
        a = random.random(N)
        b = a[i]

        i = np.arange(N)[i]
        j = np.arange(len(i))
        J = sp.csr_matrix((np.ones(len(i)), (i, j)), shape=(N,len(i))).T
        self.assertEqual(0, (b.diff(a, 'tangent') - J).nnz)
        self.assertEqual(0, (b.diff(a, 'adjoint') - J).nnz)

        c = zeros(N)
        c[i] = b
        self.assertEqual(0, (c.diff(b, 'tangent') - J.T).nnz)
        self.assertEqual(0, (c.diff(b, 'adjoint') - J.T).nnz)

        c[i] = a[i]
        self.assertEqual(0, (c.diff(a, 'tangent') - J.T * J).nnz)
        self.assertEqual(0, (c.diff(a, 'adjoint') - J.T * J).nnz)

    def test2DIndexByInt(self):
        print('testIndex')
        N = 10
        i0, i1 = [2,5,-1], [-2,3,4]
        a = random.random([N, N])
        b = a[i0,i1]

        i = np.arange(N)[i0] * N + np.arange(N)[i1]
        j = np.arange(len(i0))
        J = sp.csr_matrix((np.ones(len(i)), (i, j)), shape=(N*N,len(i))).T
        self.assertEqual(0, (b.diff(a, 'tangent') - J).nnz)
        self.assertEqual(0, (b.diff(a, 'adjoint') - J).nnz)

        c = zeros([N,N])
        c[i0,i1] = b
        self.assertEqual(0, (c.diff(b, 'tangent') - J.T).nnz)
        self.assertEqual(0, (c.diff(b, 'adjoint') - J.T).nnz)

        c[i0,i1] = a[i0,i1]
        self.assertEqual(0, (c.diff(a, 'tangent') - J.T * J).nnz)
        self.assertEqual(0, (c.diff(a, 'adjoint') - J.T * J).nnz)

    def test1DIndexBySlice(self):
        print('testIndex')
        N = 10
        a = random.random(N)
        b = a[1:]

        i = np.arange(1,N)
        j = np.arange(N-1)
        J = sp.csr_matrix((np.ones(len(i)), (i, j)), shape=(N,N-1)).T
        self.assertEqual(0, (b.diff(a, 'tangent') - J).nnz)
        self.assertEqual(0, (b.diff(a, 'adjoint') - J).nnz)

        c = zeros(N)
        c[1:] = b
        self.assertEqual(0, (c.diff(b, 'tangent') - J.T).nnz)
        self.assertEqual(0, (c.diff(b, 'adjoint') - J.T).nnz)

        c[1:] = a[1:]
        self.assertEqual(0, (c.diff(a, 'tangent') - J.T * J).nnz)
        self.assertEqual(0, (c.diff(a, 'adjoint') - J.T * J).nnz)

    def test2DIndexBySlice(self):
        print('testIndex')
        N = 4
        a = random.random([N, N])
        b0 = a[1:,:]
        b1 = a[:,1:]

        c0 = zeros([N, N])
        c1 = zeros([N, N])
        c0[1:,:] = b0
        c1[:,1:] = b1

        i = np.kron(np.arange(1,N), np.ones(N,int)) * N \
          + np.kron(np.ones(N-1,int), np.arange(N))
        j = np.arange(N * (N - 1))
        J = sp.csr_matrix((np.ones(len(i)), (i, j)), shape=(N*N, N*(N-1))).T
        self.assertEqual(0, (b0.diff(a, 'tangent') - J).nnz)
        self.assertEqual(0, (b0.diff(a, 'adjoint') - J).nnz)

        self.assertEqual(0, (c0.diff(b0, 'tangent') - J.T).nnz)
        self.assertEqual(0, (c0.diff(b0, 'adjoint') - J.T).nnz)

        c0[1:,:] = a[1:,:]
        self.assertEqual(0, (c0.diff(a, 'tangent') - J.T * J).nnz)
        self.assertEqual(0, (c0.diff(a, 'adjoint') - J.T * J).nnz)

        i = np.kron(np.arange(N), np.ones(N-1,int)) * N \
          + np.kron(np.ones(N,int), np.arange(1,N))
        j = np.arange(N * (N - 1))
        J = sp.csr_matrix((np.ones(len(i)), (i, j)), shape=(N*N, N*(N-1))).T
        self.assertEqual(0, (b1.diff(a, 'tangent') - J).nnz)
        self.assertEqual(0, (b1.diff(a, 'adjoint') - J).nnz)

        self.assertEqual(0, (c1.diff(b1, 'tangent') - J.T).nnz)
        self.assertEqual(0, (c1.diff(b1, 'adjoint') - J.T).nnz)

        c1[:,1:] = a[:,1:]
        self.assertEqual(0, (c1.diff(a, 'tangent') - J.T * J).nnz)
        self.assertEqual(0, (c1.diff(a, 'adjoint') - J.T * J).nnz)

class _OperationsTest(unittest.TestCase):
    def testAdd(self):
        print('testAdd')
        N = 1000
        a = random.random(N)
        b = random.random(N)
        c = a + b
        self.assertEqual(0, (c.diff(a, 'tangent') - sp.eye(N,N)).nnz)
        self.assertEqual(0, (c.diff(a, 'adjoint') - sp.eye(N,N)).nnz)
        self.assertEqual(0, (c.diff(b, 'tangent') - sp.eye(N,N)).nnz)
        self.assertEqual(0, (c.diff(b, 'adjoint') - sp.eye(N,N)).nnz)

    def testSub(self):
        print('testSub')
        N = 1000
        a = random.random(N)
        b = random.random(N)
        c = a - b
        self.assertEqual(0, (c.diff(a, 'tangent') - sp.eye(N,N)).nnz)
        self.assertEqual(0, (c.diff(a, 'adjoint') - sp.eye(N,N)).nnz)
        self.assertEqual(0, (c.diff(b, 'tangent') + sp.eye(N,N)).nnz)
        self.assertEqual(0, (c.diff(b, 'adjoint') + sp.eye(N,N)).nnz)

    def testMul(self):
        print('testMul')
        N = 1000
        a = random.random(N)
        b = random.random(N)
        c = a * b * 5
        self.assertEqual(0, (c.diff(a, 'tangent') - \
                5 * sp.dia_matrix((b._value, 0), (N,N))).nnz)
        self.assertEqual(0, (c.diff(a, 'adjoint') - \
                5 * sp.dia_matrix((b._value, 0), (N,N))).nnz)
        self.assertEqual(0, (c.diff(b, 'tangent') - \
                5 * sp.dia_matrix((a._value, 0), (N,N))).nnz)
        self.assertEqual(0, (c.diff(b, 'adjoint') - \
                5 * sp.dia_matrix((a._value, 0), (N,N))).nnz)

    def testDiv(self):
        print('testDiv')
        N = 10
        a = random.random(N)
        b = random.random(N)
        c = a / b / 2
        discrepancy = c.diff(a) - sp.dia_matrix((1. / b._value / 2., 0), (N,N))
        if discrepancy.nnz > 0:
            self.assertAlmostEqual(0, np.abs(discrepancy.data).max())
        discrepancy = c.diff(b) + sp.dia_matrix(((a / b**2)._value/2, 0), (N,N))
        if discrepancy.nnz > 0:
            self.assertAlmostEqual(0, np.abs(discrepancy.data).max())

    def testPow(self):
        print('testPow')
        N = 10
        a = random.random(N)
        b = 5
        c = a**b
        discrepancy = c.diff(a) - sp.dia_matrix((b * a._value**(b-1), 0), (N,N))
        if discrepancy.nnz > 0:
            self.assertAlmostEqual(0, np.abs(discrepancy.data).max())

    def testExpLog(self):
        print('testExpLog')
        N = 10
        a = random.random(N)
        c = exp(a)
        discrepancy = c.diff(a) - sp.dia_matrix((np.exp(a._value), 0), (N,N))
        if discrepancy.nnz > 0:
            self.assertAlmostEqual(0, np.abs(discrepancy.data).max())
        c = log(a)
        discrepancy = c.diff(a) - sp.dia_matrix((1 / a._value, 0), (N,N))
        if discrepancy.nnz > 0:
            self.assertAlmostEqual(0, np.abs(discrepancy.data).max())

    def testSinCos(self):
        print('testSinCos')
        N = 10
        a = random.random(N)
        b = sin(a)
        c = cos(a)
        discrepancy = b.diff(a) - sp.dia_matrix((np.cos(a._value), 0), (N,N))
        if discrepancy.nnz > 0:
            self.assertAlmostEqual(0, np.abs(discrepancy.data).max())
        discrepancy = c.diff(a) + sp.dia_matrix((np.sin(a._value), 0), (N,N))
        if discrepancy.nnz > 0:
            self.assertAlmostEqual(0, np.abs(discrepancy.data).max())

    def testSum(self):
        print('testSum')
        M, N = 4, 10
        a = random.random([M, N])
        b = sum(a, 0)
        c = sum(a, 1)

        discrepancy = b.diff(a) - sp.kron(np.ones([1,M]), sp.eye(N, N))
        if discrepancy.nnz > 0:
            self.assertAlmostEqual(0, np.abs(discrepancy.data).max())
        discrepancy = c.diff(a) - sp.kron(sp.eye(M,M), np.ones([1, N]))
        if discrepancy.nnz > 0:
            self.assertAlmostEqual(0, np.abs(discrepancy.data).max())


class _Poisson1dTest(unittest.TestCase):
    def residual(self, u, f, dx):
        res = -2 * u
        res[1:] += u[:-1]
        res[:-1] += u[1:]
        return res / dx**2 + f

    def testPoissonResidual(self):
        print('testPoissonResidual')
        #N = 40960
        N = 496
        dx = 1. / N

        f = np.random.random(N-1)
        u = np.random.random(N-1)

        t0 = time.clock()
        for i in range(100):
            res = self.residual(u, f, dx)
        print(time.clock() - t0)

        f = random.random(N-1)
        u = random.random(N-1)

        t0 = time.clock()
        res = self.residual(u, f, dx)
        print(time.clock() - t0)

        t0 = time.clock()
        dRdf_tan = res.diff(f, 'tangent')
        print('tangent', time.clock() - t0)
        t0 = time.clock()
        dRdf_adj = res.diff(f, 'adjoint')
        print('adjoint', time.clock() - t0)
        self.assertEqual((dRdf_tan - sp.eye(N-1,N-1)).nnz, 0)
        self.assertEqual((dRdf_adj - sp.eye(N-1,N-1)).nnz, 0)

        t0 = time.clock()
        dRdu_tan = res.diff(u, 'tangent')
        print('tangent', time.clock() - t0)
        t0 = time.clock()
        dRdu_adj = res.diff(u, 'adjoint')
        print('adjoint', time.clock() - t0)
        print(time.clock() - t0)

        lapl = -2 * sp.eye(N-1,N-1) \
             + sp.dia_matrix((np.ones(N-1), 1), (N-1,N-1)) \
             + sp.dia_matrix((np.ones(N-1), -1), (N-1,N-1))
        lapl /= dx**2
        self.assertEqual((dRdu_tan - lapl).nnz, 0)
        self.assertEqual((dRdu_adj - lapl).nnz, 0)


class _Poisson2dTest(unittest.TestCase):
    def residual(self, u, f, dx, dy):
        res = -(2 / dx**2 + 2 / dy**2) * u
        res[1:,:] += u[:-1,:] / dx**2
        res[:-1,:] += u[1:,:] / dx**2
        # res[:,1:] += u[:,:-1] / dy**2
        # res[:,:-1] += u[:,1:] / dy**2
        res += f
        return res

    def testPoissonResidual(self):
        print('testPoisson2DResidual')
        # N, M = 256, 512
        N, M = 3, 4
        dx, dy = 1. / N, np.inf #1. / M

        f = np.random.random((N-1, M-1))
        u = np.random.random((N-1, M-1))

        t0 = time.clock()
        for i in range(100):
            res = self.residual(u, f, dx, dy)
        print(time.clock() - t0)

        f = random.random((N-1, M-1))
        u = random.random((N-1, M-1))
        
        t0 = time.clock()
        res = self.residual(u, f, dx, dy)
        print(time.clock() - t0)

        t0 = time.clock()
        dRdf = res.diff(f, 'tangent')
        dRdf = res.diff(f, 'adjoint')
        print(time.clock() - t0)

        self.assertEqual((dRdf - sp.eye((N-1) * (M-1), (N-1) * (M-1))).nnz, 0)

        t0 = time.clock()
        dRdu_tan = res.diff(u, 'tangent')
        dRdu_adj = res.diff(u, 'adjoint')
        print(time.clock() - t0)

        lapl_i = -2 * sp.eye(N-1,N-1) \
               + sp.dia_matrix((np.ones(N-1), 1), (N-1,N-1)) \
               + sp.dia_matrix((np.ones(N-1), -1), (N-1,N-1))
        lapl_j = -2 * sp.eye(M-1,M-1) \
               + sp.dia_matrix((np.ones(M-1), 1), (M-1,M-1)) \
               + sp.dia_matrix((np.ones(M-1), -1), (M-1,M-1))
        lapl = sp.kron(lapl_i, sp.eye(M-1,M-1)) / dx**2 \
             + sp.kron(sp.eye(N-1,N-1), lapl_j) / dy**2
        self.assertEqual((dRdu_tan - lapl).nnz, 0)
        self.assertEqual((dRdu_adj - lapl).nnz, 0)

        # pylab.figure()
        # pylab.spy(lapl, marker='.')



class _Poisson3dTest(unittest.TestCase):
    def residual(self, u, f, dx, dy, dz):
        res = -(2 / dx**2 + 2 / dy**2 + 2 / dz**2) * u
        res[1:] += u[:-1] / dx**2
        res[:-1] += u[1:] / dx**2
        res[:,1:] += u[:,:-1] / dy**2
        res[:,:-1] += u[:,1:] / dy**2
        res[:,:,1:] += u[:,:,:-1] / dz**2
        res[:,:,:-1] += u[:,:,1:] / dz**2
        res += f
        return res

    def testPoissonResidual(self):
        print('testPoisson3DResidual')
        # N, M, L = 8, 24, 32
        N, M, L = 128, 24, 32
        dx, dy, dz = 1. / N, 1. / M, 1. / L

        f = np.random.random((N-1, M-1, L-1))
        u = np.random.random((N-1, M-1, L-1))
        
        t0 = time.clock()
        for i in range(100):
            res = self.residual(u, f, dx, dy, dz)
        print(time.clock() - t0)

        f = random.random((N-1, M-1, L-1))
        u = random.random((N-1, M-1, L-1))
        
        t0 = time.clock()
        res = self.residual(u, f, dx, dy, dz)
        print(time.clock() - t0)

        t0 = time.clock()
        dRdf_tan = res.diff(f, 'tangent')
        print('tangent', time.clock() - t0)
        t0 = time.clock()
        dRdf_adj = res.diff(f, 'adjoint')
        print('tangent', time.clock() - t0)
        print('adjoint', time.clock() - t0)

        self.assertEqual((dRdf_tan - sp.eye(*dRdf_tan.shape)).nnz, 0)
        self.assertEqual((dRdf_tan - sp.eye(*dRdf_adj.shape)).nnz, 0)

        t0 = time.clock()
        dRdu_tan = res.diff(u, 'tangent')
        print('tangent', time.clock() - t0)
        t0 = time.clock()
        dRdu_adj = res.diff(u, 'adjoint')
        print('adjoint', time.clock() - t0)

        lapl_i = -2 * sp.eye(N-1, N-1) \
               + sp.dia_matrix((np.ones(N-1), 1), (N-1,N-1)) \
               + sp.dia_matrix((np.ones(N-1), -1), (N-1,N-1))
        lapl_j = -2 * sp.eye(M-1, M-1) \
               + sp.dia_matrix((np.ones(M-1), 1), (M-1,M-1)) \
               + sp.dia_matrix((np.ones(M-1), -1), (M-1,M-1))
        lapl_k = -2 * sp.eye(L-1, L-1) \
               + sp.dia_matrix((np.ones(L-1), 1), (L-1,L-1)) \
               + sp.dia_matrix((np.ones(L-1), -1), (L-1,L-1))
        I_i = sp.eye(N-1, N-1)
        I_j = sp.eye(M-1, M-1)
        I_k = sp.eye(L-1, L-1)
        lapl = sp.kron(sp.kron(lapl_i, I_j), I_k) / dx**2 \
             + sp.kron(sp.kron(I_i, lapl_j), I_k) / dy**2 \
             + sp.kron(sp.kron(I_i, I_j), lapl_k) / dz**2
        self.assertEqual((dRdu_tan - lapl).nnz, 0)
        self.assertEqual((dRdu_adj - lapl).nnz, 0)

        # pylab.figure()
        # pylab.spy(lapl, marker='.')

class _Burgers1dTest(unittest.TestCase):
    def firstOrderFlux(self, u):
        u = hstack([0, u, 0])
        f = u**2 / 2
        f_max = maximum_smooth(f[1:], f[:-1])
        f_min = minimum_smooth(0, minimum_smooth(f[1:], f[:-1]))
        return lt_smooth(u[1:], u[:-1]) * f_max + \
               gt_smooth(u[1:], u[:-1]) * f_min

    def testFirstOrderResidual(self):
        print('testBurgersResidual')
        N = 4096
        dx = 1. / N
        u = random.random(N-1)
        f = self.firstOrderFlux(u)
        res = (f[1:] - f[:-1]) / dx
        self.assertTrue(res.diff(u, 'tangent').shape == (N-1,N-1))
        self.assertTrue(res.diff(u, 'adjoint').shape == (N-1,N-1))

if __name__ == '__main__':
    from numpad import *
    # _DEBUG_perturb_enable()
    unittest.main()
