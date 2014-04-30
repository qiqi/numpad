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

# --------------------- debug --------------------- #

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
    assert np.isfinite(output._base).all()
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

def base(a):
    '''
    Return the "base" of an adarray "a".  The base is a numpy.ndarray
    object containing all the data of a.
    If a is a number of a numpy.ndarray, then return a itself.
    '''
    if isinstance(a, (numbers.Number, np.ndarray, list)):
        return a
    else:
        return a._base

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
def random(*args, **kargs):
    return array(np.random.random(*args, **kargs))

@append_docstring_from_numpy
def linspace(*args, **kargs):
    return array(np.linspace(*args, **kargs))

@append_docstring_from_numpy
def arange(*args, **kargs):
    return array(np.arange(*args, **kargs))

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
        out = adarray(np.exp(x._base))
    else:
        np.exp(x._base, out._base)
        out.next_state(0, op_name='0')

    multiplier = dia_jac(np.exp(np.ravel(x._base)))
    out.next_state(multiplier, x, op_name='exp')

    if __DEBUG_MODE__:
        out._DEBUG_perturb = np.exp(x._base) * _DEBUG_perturb_retrieve(x)
        _DEBUG_perturb_verify(out)
    return out

@append_docstring_from_numpy
def sqrt(x):
    return x**(0.5)

@append_docstring_from_numpy
def sin(x, out=None):
    x = array(x)

    if out is None:
        out = adarray(np.sin(x._base))
    else:
        np.sin(x._base, out._base)
        out.next_state(0, op_name='0')
    multiplier = dia_jac(np.cos(np.ravel(x._base)))
    out.next_state(multiplier, x, op_name='sin')

    if __DEBUG_MODE__:
        out._DEBUG_perturb = np.cos(x._base) * _DEBUG_perturb_retrieve(x)
        _DEBUG_perturb_verify(out)
    return out

@append_docstring_from_numpy
def cos(x, out=None):
    x = array(x)

    if out is None:
        out = adarray(np.cos(x._base))
    else:
        np.cos(x._base, out._base)
        out.next_state(0, op_name='0')
    multiplier = dia_jac(-np.sin(np.ravel(x._base)))
    out.next_state(multiplier, x, 'cos')

    if __DEBUG_MODE__:
        out._DEBUG_perturb = -np.sin(x._base) * _DEBUG_perturb_retrieve(x)
        _DEBUG_perturb_verify(out)
    return out

@append_docstring_from_numpy
def log(x, out=None):
    x = array(x)

    if out is None:
        out = adarray(np.log(x._base))
    else:
        np.log(x._base, out._base)
        out.next_state(0, op_name='0')
    multiplier = dia_jac(1. / np.ravel(x._base))
    out.next_state(multiplier, x, 'log')

    if __DEBUG_MODE__:
        out._DEBUG_perturb = _DEBUG_perturb_retrieve(x) / x._base
        _DEBUG_perturb_verify(out)
    return out

@append_docstring_from_numpy
def tanh(x, out=None):
    x = array(x)

    if out is None:
        out = adarray(np.tanh(x._base))
    else:
        np.tanh(x._base, out._base)
        out.next_state(0, op_name='0')

    multiplier = dia_jac(1 - np.tanh(np.ravel(x._base))**2)
    out.next_state(multiplier, x, 'tanh')

    if __DEBUG_MODE__:
        out._DEBUG_perturb = _DEBUG_perturb_retrieve(x) \
                           * (1 - np.tanh(x._base)**2)
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
        adarray_a = adarray(np.array([ai._base for ai in a]))
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
    a_copy = adarray(np.copy(base(a)))
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
    a_transpose = adarray(np.transpose(a._base, axes))
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
        marker_arrays.append(len(ndarrays) * np.ones_like(a._base))
        ndarrays.append(base(a))

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
    return concatenate(adarrays, axis=0)

@append_docstring_from_numpy
def meshgrid(x, y):
    ind_xx, ind_yy = np.meshgrid(x._ind, y._ind)
    return x[ind_xx], y[ind_yy]

@append_docstring_from_numpy
def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    assert dtype is None and out is None
    a = array(a)
    sum_a = adarray(np.sum(a._base, axis, keepdims=keepdims))

    shape = np.sum(a._base, axis, keepdims=True).shape
    j = np.arange(sum_a.size).reshape(shape)
    i = np.ravel(j + np.zeros_like(a._base, int))
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
    b = adarray(np.rollaxis(a._base, axis, start))

    data = np.ones(a.size)
    j = np.ravel(a._ind)
    i = np.ravel(np.rollaxis(a._ind, axis, start))
    multiplier = csr_jac(data, i, j)

    b.next_state(multiplier, a, 'rollaxis')
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
    def __init__(self, array):
        self._base = np.asarray(base(array), np.float64)
        self._ind = np.arange(self.size).reshape(self.shape)
        self._current_state = InitialState(self)

    # def __array__(self):
    #     return self._base

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
            raise NotImplementedError()

    @property
    def size(self):
        return self._base.size
    @property
    def shape(self):
        return self._base.shape
    @property
    def ndim(self):
        return self._base.ndim
    @property
    def T(self):
        return self.transpose()

    def __len__(self):
        return self._base.__len__()

    # ------------------ object operations ----------------- #

    def copy(self):
        return copy(self)

    def transpose(self, axes=None):
        return transpose(self, axes)

    def reshape(self, shape):
        reshaped = adarray(self._base.reshape(shape))
        if self.size > 0:
            reshaped.next_state(1, self, 'reshape')
        if __DEBUG_MODE__:
            reshaped._DEBUG_perturb = self._DEBUG_perturb.reshape(shape)
        return reshaped

    def sort(self, axis=-1, kind='quicksort'):
        '''
        sort in place
        '''
        ind = np.argsort(self._base, axis, kind)
        self._base.sort(axis, kind)

        j = np.ravel(self._ind[ind])
        i = np.arange(j.size)
        multiplier = csr_jac(np.ones(j.size), i,j, shape=(j.size, self.size))
        self.next_state(multiplier, op_name='sort')

        if __DEBUG_MODE__:
            self._DEBUG_perturb = _DEBUG_perturb_retrieve(self)[ind]
            _DEBUG_perturb_verify(self)

    # ------------------ boolean operations ----------------- #

#     def __eq__(self, a):
#         return array(self._base == base(a))
# 
#     def __ne__(self, a):
#         return array(self._base != base(a))
# 
#     def __gt__(self, a):
#         return array(self._base > base(a))
#     
#     def __ge__(self, a):
#         return array(self._base >= base(a))
# 
#     def __lt__(self, a):
#         return array(self._base < base(a))
#     
#     def __le__(self, a):
#         return array(self._base <= base(a))
# 
#     def all(self):
#         return self._base.all()

    # ------------------ arithmetic operations ----------------- #

    def __add__(self, a):
        if isinstance(a, numbers.Number):
            a_p_b = adarray(self._base + a)
            a_p_b.next_state(1, self, '+')
        else:
            b = self
            a_p_b = adarray(base(a) + base(b))

            if a.shape == b.shape:
                if hasattr(a, '_base'):
                    a_p_b.next_state(1, a, '+')
                if hasattr(b, '_base'):
                    a_p_b.next_state(1, b, '+')
            else:
                # a, b, or both is "broadcasted" to fit the shape of each other
                multiplier = np.ones(a_p_b.shape)
                i = np.arange(a_p_b.size)

                if hasattr(a, '_base') and multiplier.size > 0:
                    j_a = np.ravel(a._ind_casted_to(a_p_b.shape))
                    a_multiplier = csr_jac(np.ravel(multiplier), i, j_a,
                                           shape=(a_p_b.size, a.size))
                    a_p_b.next_state(a_multiplier, a, '+')
                if hasattr(b, '_base') and multiplier.size > 0:
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
            self._base += a
        else:
            self._base += a._base
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
        neg_self = adarray(-self._base)
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
            a_x_b = adarray(self._base * a)
            a_x_b.next_state(a, self, '*')
            if __DEBUG_MODE__:
                a_x_b._DEBUG_perturb = _DEBUG_perturb_retrieve(self) * a
                _DEBUG_perturb_verify(a_x_b)
        else:
            b = self
            a_x_b = adarray(base(a) * base(b))

            if a.shape == b.shape:
                if hasattr(a, '_base'):
                    a_x_b.next_state(dia_jac(base(b).ravel()), a, '*')
                if hasattr(b, '_base'):
                    a_x_b.next_state(dia_jac(base(a).ravel()), b, '*')
            else:
                a_multiplier = np.zeros(a_x_b.shape)
                b_multiplier = np.zeros(a_x_b.shape)
                if a_multiplier.ndim:
                    a_multiplier[:] = base(b)
                else:
                    a_multiplier = base(b).copy()
                if b_multiplier.ndim:
                    b_multiplier[:] = base(a).copy()
                else:
                    b_multiplier = base(a).copy()

                i = np.arange(a_x_b.size)

                if hasattr(a, '_base') and a_x_b.size > 0:
                    j_a = np.ravel(a._ind_casted_to(a_x_b.shape))
                    a_multiplier = csr_jac(np.ravel(a_multiplier), i, j_a,
                                           shape=(a_x_b.size, a.size))
                    a_x_b.next_state(a_multiplier, a, '*')
                if hasattr(b, '_base') and a_x_b.size > 0:
                    j_b = np.ravel(b._ind_casted_to(a_x_b.shape))
                    b_multiplier = csr_jac(np.ravel(b_multiplier), i, j_b,
                                           shape=(a_x_b.size, b.size))
                    a_x_b.next_state(b_multiplier, b, '*')

            if __DEBUG_MODE__:
                a_x_b._DEBUG_perturb = _DEBUG_perturb_retrieve(self) * base(a) \
                                     + base(self) * _DEBUG_perturb_retrieve(a)
                _DEBUG_perturb_verify(a_x_b)
        return a_x_b

    def __rmul__(self, a):
        return self.__mul__(a)

    def __imul__(self, a):
        if isinstance(a, numbers.Number):
            self._base *= a
            self.next_state(a, op_name='*')
            if __DEBUG_MODE__:
                self._DEBUG_perturb *= a
                _DEBUG_perturb_verify(self)
        else:
            multiplier = dia_jac(np.ravel(a._base.copy()))
            self.next_state(multiplier, op_name='*')
            multiplier = dia_jac(np.ravel(self._base.copy()))
            self.next_state(multiplier, a, '*')
            self._base *= a._base
            if __DEBUG_MODE__:
                self._DEBUG_perturb = _DEBUG_perturb_retrieve(self) * base(a) \
                                    + base(self) * _DEBUG_perturb_retrieve(a)
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
        self_to_a = adarray(self._base ** a)
        multiplier = a * np.ravel(self._base)**(a-1)
        if multiplier.size > 0:
            multiplier[~np.isfinite(multiplier)] = 0
            multiplier = dia_jac(multiplier)
            self_to_a.next_state(multiplier, self, '**')
        if __DEBUG_MODE__:
            self_to_a._DEBUG_perturb = a * self._base**(a-1) \
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
        self_i = adarray(self._base[ind])

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
        data = np.ones(self.size)
        data[self._ind[ind]] = 0
        multiplier = dia_jac(data)
        self.next_state(multiplier, op_name='[]=0')

        self._base.__setitem__(ind, base(a))

        if hasattr(a, '_base'):
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
        return str(self._base)

    def __repr__(self):
        return 'ad' + repr(self._base)

    # ------------------ differentiation ------------------ #

    def diff(self, u, mode='auto'):
        if mode == 'auto':
            if u.size < self.size:
                mode = 'tangent'
            else:
                mode = 'adjoint'

        if mode == 'tangent':
            derivative = diff_tangent(self._current_state, u._initial_state)
        elif mode == 'adjoint':
            derivative = diff_adjoint(self._current_state, u._initial_state)
        else:
            raise NotImplementedError()

        return derivative


# =========================================================== #
#                                                             #
#                         unittests                           #
#                                                             #
# =========================================================== #

class _ManipulationTest(unittest.TestCase):
    def testArray(self):
        N = 10
        a = random(N)
        b = random(N)
        c = random(N)
        d = random(N)
        e = array([[a, b], [c, d]])

        I, O = sp.eye(N,N), sp.eye(N,N) * 0
        self.assertEqual(0, (e.diff(a, 'tangent') - sp.vstack([I,O,O,O])).nnz)
        self.assertEqual(0, (e.diff(a, 'adjoint') - sp.vstack([I,O,O,O])).nnz)
        self.assertEqual(0, (e.diff(b, 'tangent') - sp.vstack([O,I,O,O])).nnz)
        self.assertEqual(0, (e.diff(b, 'adjoint') - sp.vstack([O,I,O,O])).nnz)
        self.assertEqual(0, (e.diff(c, 'tangent') - sp.vstack([O,O,I,O])).nnz)
        self.assertEqual(0, (e.diff(c, 'adjoint') - sp.vstack([O,O,I,O])).nnz)
        self.assertEqual(0, (e.diff(d, 'tangent') - sp.vstack([O,O,O,I])).nnz)
        self.assertEqual(0, (e.diff(d, 'adjoint') - sp.vstack([O,O,O,I])).nnz)

    def testDot(self):
        N = 20
        a = random((10, N))
        b = random((N, 30))
        c = dot(a, b)

        c_diff_b = c.diff(b)
        discrepancy = c_diff_b - sp.kron(a._base, sp.eye(c.shape[1]))
        if discrepancy.nnz > 0:
            self.assertAlmostEqual(0, np.abs(discrepancy.data).max())

        c_diff_a = c.diff(a)
        discrepancy = c_diff_a - sp.kron(sp.eye(c.shape[0]), b.T._base)
        if discrepancy.nnz > 0:
            self.assertAlmostEqual(0, np.abs(discrepancy.data).max())

    def testTranspose(self):
        N = 10
        a = random(N)
        b = random(N)
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
    def testIndex(self):
        N = 10
        i = [2,5,-1]
        a = random(N)
        b = a[i]

        i = np.arange(N)[i]
        j = np.arange(len(i))
        J = sp.csr_matrix((np.ones(len(i)), (i, j)), shape=(N,len(i))).T
        self.assertEqual(0, (b.diff(a, 'tangent') - J).nnz)
        self.assertEqual(0, (b.diff(a, 'adjoint') - J).nnz)


class _OperationsTest(unittest.TestCase):
    def testAdd(self):
        N = 1000
        a = random(N)
        b = random(N)
        c = a + b
        self.assertEqual(0, (c.diff(a, 'tangent') - sp.eye(N,N)).nnz)
        self.assertEqual(0, (c.diff(a, 'adjoint') - sp.eye(N,N)).nnz)
        self.assertEqual(0, (c.diff(b, 'tangent') - sp.eye(N,N)).nnz)
        self.assertEqual(0, (c.diff(b, 'adjoint') - sp.eye(N,N)).nnz)

    def testSub(self):
        N = 1000
        a = random(N)
        b = random(N)
        c = a - b
        self.assertEqual(0, (c.diff(a, 'tangent') - sp.eye(N,N)).nnz)
        self.assertEqual(0, (c.diff(a, 'adjoint') - sp.eye(N,N)).nnz)
        self.assertEqual(0, (c.diff(b, 'tangent') + sp.eye(N,N)).nnz)
        self.assertEqual(0, (c.diff(b, 'adjoint') + sp.eye(N,N)).nnz)

    def testMul(self):
        N = 1000
        a = random(N)
        b = random(N)
        c = a * b * 5
        self.assertEqual(0, (c.diff(a, 'tangent') - \
                5 * sp.dia_matrix((b._base, 0), (N,N))).nnz)
        self.assertEqual(0, (c.diff(a, 'adjoint') - \
                5 * sp.dia_matrix((b._base, 0), (N,N))).nnz)
        self.assertEqual(0, (c.diff(b, 'tangent') - \
                5 * sp.dia_matrix((a._base, 0), (N,N))).nnz)
        self.assertEqual(0, (c.diff(b, 'adjoint') - \
                5 * sp.dia_matrix((a._base, 0), (N,N))).nnz)

    def testDiv(self):
        N = 10
        a = random(N)
        b = random(N)
        c = a / b / 2
        discrepancy = c.diff(a) - sp.dia_matrix((1. / b._base / 2., 0), (N,N))
        if discrepancy.nnz > 0:
            self.assertAlmostEqual(0, np.abs(discrepancy.data).max())
        discrepancy = c.diff(b) + sp.dia_matrix(((a / b**2)._base/2, 0), (N,N))
        if discrepancy.nnz > 0:
            self.assertAlmostEqual(0, np.abs(discrepancy.data).max())

    def testPow(self):
        N = 10
        a = random(N)
        b = 5
        c = a**b
        discrepancy = c.diff(a) - sp.dia_matrix((b * a._base**(b-1), 0), (N,N))
        if discrepancy.nnz > 0:
            self.assertAlmostEqual(0, np.abs(discrepancy.data).max())

    def testExpLog(self):
        N = 10
        a = random(N)
        c = exp(a)
        discrepancy = c.diff(a) - sp.dia_matrix((np.exp(a._base), 0), (N,N))
        if discrepancy.nnz > 0:
            self.assertAlmostEqual(0, np.abs(discrepancy.data).max())
        c = log(a)
        discrepancy = c.diff(a) - sp.dia_matrix((1 / a._base, 0), (N,N))
        if discrepancy.nnz > 0:
            self.assertAlmostEqual(0, np.abs(discrepancy.data).max())

    def testSinCos(self):
        N = 10
        a = random(N)
        b = sin(a)
        c = cos(a)
        discrepancy = b.diff(a) - sp.dia_matrix((np.cos(a._base), 0), (N,N))
        if discrepancy.nnz > 0:
            self.assertAlmostEqual(0, np.abs(discrepancy.data).max())
        discrepancy = c.diff(a) + sp.dia_matrix((np.sin(a._base), 0), (N,N))
        if discrepancy.nnz > 0:
            self.assertAlmostEqual(0, np.abs(discrepancy.data).max())

    def testSum(self):
        M, N = 4, 10
        a = random([M, N])
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
        print()
        #N = 40960
        N = 496
        dx = 1. / N

        f = np.random.random(N-1)
        u = np.random.random(N-1)

        t0 = time.clock()
        for i in range(100):
            res = self.residual(u, f, dx)
        print(time.clock() - t0)

        f = random(N-1)
        u = random(N-1)

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
        res[:,1:] += u[:,:-1] / dy**2
        res[:,:-1] += u[:,1:] / dy**2
        res += f
        return res

    def testPoissonResidual(self):
        print()
        N, M = 256, 512
        # N, M = 256, 12
        dx, dy = 1. / N, 1. / M

        f = np.random.random((N-1, M-1))
        u = np.random.random((N-1, M-1))

        t0 = time.clock()
        for i in range(100):
            res = self.residual(u, f, dx, dy)
        print(time.clock() - t0)

        f = random((N-1, M-1))
        u = random((N-1, M-1))
        
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
        print()
        # N, M, L = 8, 24, 32
        N, M, L = 128, 24, 32
        dx, dy, dz = 1. / N, 1. / M, 1. / L

        f = np.random.random((N-1, M-1, L-1))
        u = np.random.random((N-1, M-1, L-1))
        
        t0 = time.clock()
        for i in range(100):
            res = self.residual(u, f, dx, dy, dz)
        print(time.clock() - t0)

        f = random((N-1, M-1, L-1))
        u = random((N-1, M-1, L-1))
        
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
        N = 4096
        dx = 1. / N
        u = random(N-1)
        f = self.firstOrderFlux(u)
        res = (f[1:] - f[:-1]) / dx
        self.assertTrue(res.diff(u, 'tangent').shape == (N-1,N-1))
        self.assertTrue(res.diff(u, 'adjoint').shape == (N-1,N-1))

if __name__ == '__main__':
    # _DEBUG_mode()
    unittest.main()
