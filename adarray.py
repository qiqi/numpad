import time
import unittest
import numbers
import pylab
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

# __all__ = ['base', 'zeros', 'ones', 'random', 'linspace', 'loadtxt',
# 'sigmoid', 'gt_smooth', 'lt_smooth', 'maximum_smooth', 'minimum_smooth',
# ]

# --------------------- debug --------------------- #

__DEBUG_MODE__ = False
__DEBUG_TOL__ = None
__DEBUG_SEED_ARRAYS__ = []

def _DEBUG_enable(enable=True, tolerance=None):
    '''
    Turn __DEBUG_MODE__ on.
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

def _DEBUG_verify(output, message=''):
    '''
    Verify a dependent variable (output) against random perturbations.
    '''
    assert np.isfinite(output._base).all()
    out_perturb = np.zeros(output.size)
    for var, var_perturb in __DEBUG_SEED_ARRAYS__:
        J = output.diff(var)
        if J is not 0:
            out_perturb += J * var_perturb
    error_norm = np.linalg.norm(out_perturb - np.ravel(output._DEBUG_perturb))
    print('_DEBUG_verify ', message, ': ', error_norm)
    if __DEBUG_TOL__:
        assert error_norm < __DEBUG_TOL__

def _DEBUG_new_perturb(var):
    global __DEBUG_MODE__, __DEBUG_SEED_ARRAYS__
    if __DEBUG_MODE__:
        var._DEBUG_perturb = np.random.random(var.shape)
        __DEBUG_SEED_ARRAYS__.append((var, np.ravel(var._DEBUG_perturb.copy())))
    return var

def _DEBUG_perturb(var):
    if hasattr(var, '_DEBUG_perturb'):
        return var._DEBUG_perturb
    elif isinstance(var, adarray):
        return np.zeros(var.shape)
    else:
        return np.zeros(np.asarray(var).shape)

# --------------------- utilities --------------------- #

def base(a):
    if isinstance(a, (numbers.Number, np.ndarray, list)):
        return a
    else:
        return a._base

# --------------------- adarray construction --------------------- #

def zeros(shape):
    new_array = adarray(np.zeros(shape))
    _DEBUG_new_perturb(new_array)
    return new_array

def ones(shape):
    new_array = adarray(np.ones(shape))
    _DEBUG_new_perturb(new_array)
    return new_array

def random(shape):
    new_array = adarray(np.random.random(shape))
    _DEBUG_new_perturb(new_array)
    return new_array

def linspace(start, stop, num=50, endpoint=True, retstep=False):
    new_array = adarray(np.linspace(start, stop, num, endpoint, retstep))
    _DEBUG_new_perturb(new_array)
    return new_array

def loadtxt(fname, dtype=float, comments='#', delimiter=None,
        converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0):
    return array(np.loadtxt(fname, dtype, comments, delimiter, converters,
        skiprows, usecols, unpack, ndmin))

# --------------------- algebraic functions --------------------- #

# def maximum(a, b):
#     a_gt_b = a > b
#     return a * a_gt_b + b * (1. - a_gt_b)
# 
# def minimum(a, b):
#     a_gt_b = a > b
#     return b * a_gt_b + a * (1. - a_gt_b)

def sigmoid(x):
    a = exp(-2 * x)
    return 1 / (1 + a)

def gt_smooth(a, b, c=0.1):
    return sigmoid((a - b) / c)

def lt_smooth(a, b, c=0.1):
    return sigmoid((b - a) / c)

def maximum_smooth(a, b, c=0.1):
    a_gt_b = gt_smooth(a, b, c)
    return a * a_gt_b + b * (1. - a_gt_b)

def minimum_smooth(a, b, c=0.1):
    return -maximum_smooth(-a, -b, c)

def exp(x, out=None):
    if isinstance(x, (numbers.Number, np.ndarray)):
        return np.exp(x, out)
    else:
        if out is None:
            out = adarray(np.exp(x._base))
        else:
            np.exp(x._base, out._base)
            out.self_ops(0)
        multiplier = sp.dia_matrix((np.exp(np.ravel(x._base)), 0),
                                   (x.size, x.size))
        out.add_ops(x, multiplier)

        if __DEBUG_MODE__:
            out._DEBUG_perturb = np.exp(x._base) * _DEBUG_perturb(x)
            _DEBUG_verify(out)
        return out

def sqrt(x):
    return x**(0.5)

def sin(x, out=None):
    if isinstance(x, (numbers.Number, np.ndarray)):
        return np.sin(x, out)
    else:
        if out is None:
            out = adarray(np.sin(x._base))
        else:
            np.sin(x._base, out._base)
            out.self_ops(0)
        multiplier = sp.dia_matrix((np.cos(np.ravel(x._base)), 0),
                                   (x.size, x.size))
        out.add_ops(x, multiplier)

        if __DEBUG_MODE__:
            out._DEBUG_perturb = np.cos(x._base) * _DEBUG_perturb(x)
            _DEBUG_verify(out)
        return out

def cos(x, out=None):
    if isinstance(x, (numbers.Number, np.ndarray)):
        return np.cos(x, out)
    else:
        if out is None:
            out = adarray(np.cos(x._base))
        else:
            np.cos(x._base, out._base)
            out.self_ops(0)
        multiplier = sp.dia_matrix((-np.sin(np.ravel(x._base)), 0),
                                   (x.size, x.size))
        out.add_ops(x, multiplier)

        if __DEBUG_MODE__:
            out._DEBUG_perturb = -np.sin(x._base) * _DEBUG_perturb(x)
            _DEBUG_verify(out)
        return out

def log(x, out=None):
    if isinstance(x, (numbers.Number, np.ndarray)):
        return np.log(x, out)
    else:
        if out is None:
            out = adarray(np.log(x._base))
        else:
            np.log(x._base, out._base)
            out.self_ops(0)
        multiplier = sp.dia_matrix((1. / np.ravel(x._base), 0),
                                   (x.size, x.size))
        out.add_ops(x, multiplier)

        if __DEBUG_MODE__:
            out._DEBUG_perturb = _DEBUG_perturb(x) / x._base
            _DEBUG_verify(out)
        return out

# ------------------ copy, stack, transpose operations ------------------- #

def array(a):
    if isinstance(a, adarray):
        return a
    elif isinstance(a, (numbers.Number, np.ndarray)):
        return adarray(a)
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
            multiplier = sp.csr_matrix((data, (i_data, j_data)), shape=shape)
            adarray_a.add_ops(a[i], multiplier)

        if __DEBUG_MODE__:
            _DEBUG_perturb_list = []
            for i in range(len(a)):
                _DEBUG_perturb_list.append(_DEBUG_perturb(a[i]))
            adarray_a._DEBUG_perturb = np.array(_DEBUG_perturb_list)
            _DEBUG_verify(adarray_a)

        return adarray_a
        
def ravel(a):
    a = array(a)
    return a.reshape((a.size,))

def copy(a):
    a_copy = adarray(np.copy(base(a)))
    if isinstance(a, adarray):
        a_copy.add_ops(a, 1)
        if __DEBUG_MODE__:
            a_copy._DEBUG_perturb = _DEBUG_perturb(a).copy()
            _DEBUG_verify(a_copy)
    else:
        assert isinstance(a, np.ndarray)
    return a_copy

def transpose(a, axes=None):
    a = array(a)
    a_transpose = adarray(np.transpose(a._base, axes))
    i = np.arange(a.size).reshape(a.shape)
    j = np.transpose(i, axes)
    data = np.ones(i.size)
    multiplier = sp.csr_matrix((data, (np.ravel(i), np.ravel(j))))
    a_transpose.add_ops(a, multiplier)
    if __DEBUG_MODE__:
        a_transpose._DEBUG_perturb = _DEBUG_perturb(a).T
        _DEBUG_verify(a_transpose)
    return a_transpose

def hstack(adarrays):
    ndarrays = []
    components, marker_arrays = [], []
    for array in adarrays:
        ndarrays.append(base(array))
        if isinstance(array, (numbers.Number, np.ndarray)):
            marker_arrays.append(np.zeros_like(array))
        else:
            components.append(array)
            marker_arrays.append(len(components) * np.ones_like(array._base))
    stacked_array = adarray(np.hstack(ndarrays))
    marker = np.ravel(np.hstack(marker_arrays))
    # marker now contains integers. 0 means nothing, 1 means the 1st component
    for i_component, component in enumerate(components):
        i = (marker == i_component + 1).nonzero()[0]
        j = np.arange(i.size)
        data = np.ones(i.size, int)
        multiplier = sp.csr_matrix((data, (i, j)), shape=(marker.size, i.size))
        stacked_array.add_ops(component, multiplier)

    if __DEBUG_MODE__:
        _DEBUG_perturb_list = []
        for array in adarrays:
            _DEBUG_perturb_list.append(_DEBUG_perturb(array))
        stacked_array._DEBUG_perturb = np.hstack(_DEBUG_perturb_list)
        _DEBUG_verify(stacked_array)
    return stacked_array

def vstack(adarrays):
    return hstack([a.T for a in adarrays]).T

def meshgrid(x, y):
    ind_xx, ind_yy = np.meshgrid(x._ind, y._ind)
    return x[ind_xx], y[ind_yy]

def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    assert dtype is None and out is None
    a = array(a)
    sum_a = adarray(np.sum(a._base, axis, keepdims=keepdims))

    shape = np.sum(a._base, axis, keepdims=True).shape
    j = np.arange(sum_a.size).reshape(shape)
    i = np.ravel(j + np.zeros_like(a._base, int))
    j = np.ravel(a._ind)
    data = np.ones(i.size, int)
    multiplier = sp.csr_matrix((data, (i, j)), shape=(sum_a.size, a.size))
    sum_a.add_ops(a, multiplier)

    if __DEBUG_MODE__:
        sum_a._DEBUG_perturb = np.sum(a._DEBUG_perturb, axis, keepdims=keepdims)
        _DEBUG_verify(stacked_array)
    return sum_a

def mean(a, axis=None, dtype=None, out=None, keepdims=False):
    sum_a = sum(a, axis, dtype, out, keepdims)
    return sum_a * (float(sum_a.size) / a.size)

# ===================== the adarray class ====================== #

class adarray:
    def __init__(self, array):
        self._base = np.asarray(base(array), np.float64)
        self._ops = []
        self._ind = np.arange(self.size).reshape(self.shape)

    def _ind_casted_to(self, shape):
        ind = np.zeros(shape, dtype=int)
        ind[:] = self._ind
        return ind

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

    # -------------------- ops management ----------------- #
    def i_ops(self):
        '''
        Total number of dependent operations
        '''
        return len(self._ops)

    def add_ops(self, other, multiplier):
        '''
        Add graph link pointing to other -- x += multiplier * other
        '''
        if multiplier is not 0:
            if not isinstance(multiplier, numbers.Number):
                assert multiplier.shape == (self.size, other.size)
            self._ops.append((other, other.i_ops(), multiplier))

    def self_ops(self, multiplier):
        '''
        Add graph link pointing to itself -- x *= multiplier
        '''
        if not isinstance(multiplier, numbers.Number):
            assert multiplier.shape == (self.size, self.size)
        self._ops.append((multiplier,))

    # ------------------ object operations ----------------- #

    def copy(self):
        return copy(self)

    def transpose(self, axes=None):
        return transpose(self, axes)

    def reshape(self, shape):
        reshaped = adarray(self._base.reshape(shape))
        reshaped.add_ops(self, sp.eye(self.size,self.size))
        return reshaped

    def sort(self, axis=-1, kind='quicksort'):
        '''
        sort in place
        '''
        ind = np.argsort(self._base, axis, kind)
        self._base.sort(axis, kind)

        j = np.ravel(self._ind[ind])
        i = np.arange(j.size)
        multiplier = sp.csr_matrix((np.ones(j.size), (i,j)),
                                   shape=(j.size, self.size))
        self.self_ops(multiplier)

        if __DEBUG_MODE__:
            self._DEBUG_perturb = _DEBUG_perturb(self)[ind]
            _DEBUG_verify(self)

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
        if isinstance(a, (numbers.Number, np.ndarray)):
            self_plus_a = adarray(self._base + a)
            self_plus_a.add_ops(self, 1)
        else:
            self_plus_a = adarray(self._base + a._base)
            self_plus_a.add_ops(self, 1)
            self_plus_a.add_ops(a, 1)

        if __DEBUG_MODE__:
            self_plus_a._DEBUG_perturb = _DEBUG_perturb(self) \
                                       + _DEBUG_perturb(a)
            _DEBUG_verify(self_plus_a)
        return self_plus_a

    def __radd__(self, a):
        return self.__add__(a)

    def __iadd__(self, a):
        if isinstance(a, (numbers.Number, np.ndarray)):
            self._base += a
        else:
            self._base += a._base
            self.add_ops(a, 1)

        if __DEBUG_MODE__:
            self._DEBUG_perturb += _DEBUG_perturb(a)
            _DEBUG_verify(self)
        return self

    def __neg__(self):
        neg_self = adarray(-self._base)
        neg_self.add_ops(self, -1)
        if __DEBUG_MODE__:
            neg_self._DEBUG_perturb = -_DEBUG_perturb(self)
            _DEBUG_verify(neg_self)
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
            a_x_b.add_ops(self, a)
            if __DEBUG_MODE__:
                a_x_b._DEBUG_perturb = _DEBUG_perturb(self) * a
                _DEBUG_verify(a_x_b)
        else:
            b = self
            if a.size > b.size:
                a, b = b, a
            a_x_b = adarray(base(a) * base(b))

            a_multiplier = np.zeros(a_x_b.shape)
            b_multiplier = np.zeros(a_x_b.shape)
            a_multiplier[:] = base(b)
            b_multiplier[:] = base(a)

            i = np.arange(a_x_b.size)
            j_a = np.ravel(a._ind_casted_to(a_x_b.shape))
            j_b = np.ravel(b._ind_casted_to(a_x_b.shape))
            a_multiplier = sp.csr_matrix((np.ravel(a_multiplier), (i, j_a)),
                                         shape=(a_x_b.size, a.size))
            b_multiplier = sp.csr_matrix((np.ravel(b_multiplier), (i, j_b)),
                                         shape=(a_x_b.size, b.size))

            if not isinstance(a, np.ndarray): a_x_b.add_ops(a, a_multiplier)
            if not isinstance(b, np.ndarray): a_x_b.add_ops(b, b_multiplier)

            if __DEBUG_MODE__:
                a_x_b._DEBUG_perturb = _DEBUG_perturb(self) * base(a) \
                                     + base(self) * _DEBUG_perturb(a)
                _DEBUG_verify(a_x_b)
        return a_x_b

    def __rmul__(self, a):
        return self.__mul__(a)

    def __imul__(self, a):
        if isinstance(a, numbers.Number):
            self._base *= a
            self.self_ops(a)
            if __DEBUG_MODE__:
                self._DEBUG_perturb *= a
                _DEBUG_verify(self)
        else:
            self._base *= a._base
            multiplier = sp.dia_matrix((np.ravel(a._base), 0), (a.size,a.size))
            self.self_ops(multiplier)
            multiplier = sp.dia_matrix((np.ravel(self._base), 0),
                                       (self.size, self.size))
            self.add_ops(a, multiplier)
            if __DEBUG_MODE__:
                self._DEBUG_perturb = _DEBUG_perturb(self) * base(a) \
                                    + base(self) * _DEBUG_perturb(a)
                _DEBUG_verify(self)
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
        assert isinstance(a, numbers.Number)
        self_to_a = adarray(self._base ** a)
        multiplier = a * np.ravel(self._base)**(a-1)
        multiplier[~np.isfinite(multiplier)] = 0
        multiplier = sp.dia_matrix((multiplier, 0), (self.size,self.size))
        self_to_a.add_ops(self, multiplier)
        if __DEBUG_MODE__:
            self_to_a._DEBUG_perturb = a * self._base**(a-1) \
                                     * _DEBUG_perturb(self)
            _DEBUG_verify(self_to_a)
        return self_to_a
    
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
            multiplier = sp.csr_matrix((np.ones(j.size), (i,j)),
                                       shape=(j.size, self.size))
            self_i.add_ops(self, multiplier)

        if __DEBUG_MODE__:
            self_i._DEBUG_perturb = _DEBUG_perturb(self)[ind]
            _DEBUG_verify(self_i)
        return self_i

    def __setitem__(self, ind, a):
        data = np.ones(self.size)
        data[self._ind[ind]] = 0
        multiplier = sp.dia_matrix((data, 0), (data.size, data.size))
        self.self_ops(multiplier)

        self._base.__setitem__(ind, base(a))

        if hasattr(a, '_base'):
            i = self._ind[ind]
            if i.size > 0:
                j = a._ind_casted_to(i.shape)
                i, j = np.ravel(i), np.ravel(j)
                multiplier = sp.csr_matrix((np.ones(j.size), (i,j)),
                                           shape=(self.size, a.size))
                self.add_ops(a, multiplier)

        if __DEBUG_MODE__:
            self._DEBUG_perturb[ind] = _DEBUG_perturb(a)
            _DEBUG_verify(self)

    # ------------------ str, repr ------------------ #

    def __str__(self):
        return str(self._base)

    def __repr__(self):
        return 'ad' + repr(self._base)

    # ------------------ differentiation ------------------ #

    def diff(self, u):
        _clear_tmp_product(self, self.i_ops())
        return _diff_recurse(self, u, self.i_ops())

# ------------------ recursive functions for differentiation --------------- #
def _clear_tmp_product(f, i_f_ops):
    if hasattr(f, '_tmp_product') and i_f_ops in f._tmp_product:
        del f._tmp_product[i_f_ops]
        if not f._tmp_product:   # empty
            del f._tmp_product

        if i_f_ops > 0:
            op = f._ops[i_f_ops - 1]
            if len(op) == 1:  # self operation
                _clear_tmp_product(f, i_f_ops - 1)
            else:
                other, i_other_ops, multiplier = op
                _clear_tmp_product(other, i_other_ops)
                _clear_tmp_product(f, i_f_ops - 1)

def _diff_recurse(f, u, i_f_ops):
    def multiply_ops(op0, op1):
        if op0 is 0 or op1 is 0:
            return 0
        else:
            return op0 * op1

    def add_ops(op0, op1):
        if op0 is 0:
            return op1
        elif op1 is 0:
            return op0
        else:
            return op0 + op1

    # function starts here
    if not hasattr(f, '_tmp_product'):
        f._tmp_product = {}
    elif i_f_ops in f._tmp_product:
        return f._tmp_product[i_f_ops]

    if i_f_ops == 0:  # I got to the bottom
        if u is f:
            product = sp.eye(u.size, u.size)
        else:
            product = 0
    else:
        op = f._ops[i_f_ops - 1]
        if len(op) == 1:  # self operation
            multiplier = op[0]
            multiplier1 = _diff_recurse(f, u, i_f_ops - 1)
            product = multiply_ops(multiplier, multiplier1)
        else:
            other, i_other_ops, multiplier = op
            multiplier1 = _diff_recurse(other, u, i_other_ops)
            other_diff = multiply_ops(multiplier, multiplier1)
            this_diff = _diff_recurse(f, u, i_f_ops - 1)
            product = add_ops(this_diff, other_diff)

    f._tmp_product[i_f_ops] = product
    return product


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
        self.assertEqual(0, (e.diff(a) - sp.vstack([I, O, O, O])).nnz)
        self.assertEqual(0, (e.diff(b) - sp.vstack([O, I, O, O])).nnz)
        self.assertEqual(0, (e.diff(c) - sp.vstack([O, O, I, O])).nnz)
        self.assertEqual(0, (e.diff(d) - sp.vstack([O, O, O, I])).nnz)

    def testTranspose(self):
        N = 10
        a = random(N)
        b = random(N)
        c = transpose([a, b])

        i, j = np.arange(N) * 2, np.arange(N)
        c_diff_a = sp.csr_matrix((np.ones(N), (i,j)), shape=(2*N, N))
        i, j = np.arange(N) * 2 + 1, np.arange(N)
        c_diff_b = sp.csr_matrix((np.ones(N), (i,j)), shape=(2*N, N))
        self.assertEqual(0, (c.diff(a) - c_diff_a).nnz)
        self.assertEqual(0, (c.diff(b) - c_diff_b).nnz)


class _IndexingTest(unittest.TestCase):
    def testIndex(self):
        N = 10
        i = [2,5,-1]
        a = random(N)
        b = a[i]

        i = np.arange(N)[i]
        j = np.arange(len(i))
        J = sp.csr_matrix((np.ones(len(i)), (i, j)), shape=(N,len(i))).T
        self.assertEqual(0, (b.diff(a) - J).nnz)


class _OperationsTest(unittest.TestCase):
    def testAdd(self):
        N = 1000
        a = random(N)
        b = random(N)
        c = a + b
        self.assertEqual(0, (c.diff(a) - sp.eye(N,N)).nnz)
        self.assertEqual(0, (c.diff(b) - sp.eye(N,N)).nnz)

    def testSub(self):
        N = 1000
        a = random(N)
        b = random(N)
        c = a - b
        self.assertEqual(0, (c.diff(a) - sp.eye(N,N)).nnz)
        self.assertEqual(0, (c.diff(b) + sp.eye(N,N)).nnz)

    def testMul(self):
        N = 1000
        a = random(N)
        b = random(N)
        c = a * b * 5
        self.assertEqual(0,
                (c.diff(a) - 5 * sp.dia_matrix((b._base, 0), (N,N))).nnz)
        self.assertEqual(0,
                (c.diff(b) - 5 * sp.dia_matrix((a._base, 0), (N,N))).nnz)

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
        discrepancy = c.diff(a) - sp.dia_matrix((exp(a._base), 0), (N,N))
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
        discrepancy = b.diff(a) - sp.dia_matrix((cos(a._base), 0), (N,N))
        if discrepancy.nnz > 0:
            self.assertAlmostEqual(0, np.abs(discrepancy.data).max())
        discrepancy = c.diff(a) + sp.dia_matrix((sin(a._base), 0), (N,N))
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
        dRdf = res.diff(f)
        print(time.clock() - t0)
        self.assertEqual((dRdf - sp.eye(N-1,N-1)).nnz, 0)

        t0 = time.clock()
        dRdu = res.diff(u)
        print(time.clock() - t0)

        lapl = -2 * sp.eye(N-1,N-1) \
             + sp.dia_matrix((np.ones(N-1), 1), (N-1,N-1)) \
             + sp.dia_matrix((np.ones(N-1), -1), (N-1,N-1))
        lapl /= dx**2
        self.assertEqual((dRdu - lapl).nnz, 0)


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
        dRdf = res.diff(f)
        print(time.clock() - t0)

        self.assertEqual((dRdf - sp.eye((N-1) * (M-1), (N-1) * (M-1))).nnz, 0)

        t0 = time.clock()
        dRdu = res.diff(u)
        print(time.clock() - t0)

        lapl_i = -2 * sp.eye(N-1,N-1) \
               + sp.dia_matrix((np.ones(N-1), 1), (N-1,N-1)) \
               + sp.dia_matrix((np.ones(N-1), -1), (N-1,N-1))
        lapl_j = -2 * sp.eye(M-1,M-1) \
               + sp.dia_matrix((np.ones(M-1), 1), (M-1,M-1)) \
               + sp.dia_matrix((np.ones(M-1), -1), (M-1,M-1))
        lapl = sp.kron(lapl_i, sp.eye(M-1,M-1)) / dx**2 \
             + sp.kron(sp.eye(N-1,N-1), lapl_j) / dy**2
        self.assertEqual((dRdu - lapl).nnz, 0)

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
        dRdf = res.diff(f)
        print(time.clock() - t0)

        self.assertEqual((dRdf - sp.eye(*dRdf.shape)).nnz, 0)

        t0 = time.clock()
        dRdu = res.diff(u)
        print(time.clock() - t0)

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
        self.assertEqual((dRdu - lapl).nnz, 0)

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
        self.assertTrue(res.diff(u).shape == (N-1,N-1))

if __name__ == '__main__':
    # _DEBUG_mode()
    unittest.main()
