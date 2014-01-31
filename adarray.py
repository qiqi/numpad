import time
import unittest
import numbers
import pylab
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

# --------------------- debug --------------------- #

__DEBUG_MODE__ = False
__DEBUG_TOL__ = None
__DEBUG_SEED_ARRAYS__ = []

def _DEBUG_mode(mode=True, tolerance=None):
    global __DEBUG_MODE__, __DEBUG_TOL__
    assert(isinstance(mode, bool))
    __DEBUG_MODE__ = mode
    __DEBUG_TOL__ = tolerance

def _DEBUG_check(output, message=''):
    out_perturb = np.zeros(output.size)
    for var, var_perturb in __DEBUG_SEED_ARRAYS__:
        J = output.diff(var)
        if J is not 0:
            out_perturb += J * var_perturb
    error_norm = np.linalg.norm(out_perturb - np.ravel(output._DEBUG_perturb))
    print('_DEBUG_check ', message, ': ', error_norm)
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

def maximum(a, b):
    a_gt_b = a > b
    return a * a_gt_b + b * (1. - a_gt_b)

def minimum(a, b):
    a_gt_b = a > b
    return b * a_gt_b + a * (1. - a_gt_b)

def exp(x, out=None):
    if isinstance(x, (numbers.Number, np.ndarray)):
        return np.exp(x, out)
    else:
        if out is None:
            out = adarray(np.exp(x._base))
        else:
            np.exp(x._base, out._base)
            out.self_ops(0)
        multiplier = sp.dia_matrix((np.exp(x._base), 0), (x.size, x.size))
        out.add_ops(x, multiplier)

        if __DEBUG_MODE__:
            out._DEBUG_perturb = np.exp(x._base) * _DEBUG_perturb(x)
            _DEBUG_check(out)
        return out

# ------------------ copy, stack, transpose operations ------------------- #

def array(a):
    if isinstance(a, adarray):
        return a
    elif isinstance(a, np.ndarray):
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
            _DEBUG_check(adarray_a)

        return adarray_a
        
def ravel(a):
    return a.reshape((a.size,))

def copy(a):
    a_copy = adarray(np.copy(base(a)))
    if isinstance(a, adarray):
        a_copy.add_ops(a, 1)
        if __DEBUG_MODE__:
            a_copy._DEBUG_perturb = _DEBUG_perturb(a).copy()
            _DEBUG_check(a_copy)
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
        _DEBUG_check(a_transpose)
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
        _DEBUG_check(stacked_array)
    return stacked_array


# ===================== the adarray class ====================== #

class adarray:
    def __init__(self, array):
        self._base = np.asarray(base(array), np.float64)
        self._ops = []
        self._ind = np.arange(self.size).reshape(self.shape)

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
            _DEBUG_check(self)

    # ------------------ boolean operations ----------------- #

    def __eq__(self, a):
        return array(self._base == base(a))

    def __ne__(self, a):
        return array(self._base != base(a))

    def __gt__(self, a):
        return array(self._base > base(a))
    
    def __ge__(self, a):
        return array(self._base >= base(a))

    def __lt__(self, a):
        return array(self._base < base(a))
    
    def __le__(self, a):
        return array(self._base <= base(a))

    def all(self):
        return self._base.all()

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
            _DEBUG_check(self_plus_a)
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
            _DEBUG_check(self)
        return self

    def __neg__(self):
        neg_self = adarray(-self._base)
        neg_self.add_ops(self, -1)
        if __DEBUG_MODE__:
            neg_self._DEBUG_perturb = -_DEBUG_perturb(self)
            _DEBUG_check(neg_self)
        return neg_self
        
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
                _DEBUG_check(a_x_b)
        else:
            b = self
            if a.size > b.size:
                a, b = b, a
            a_x_b = adarray(base(a) * base(b))

            a_multiplier = np.asarray(np.ravel(base(b)), float)
            b_multiplier = np.asarray(np.ravel(base(a)), float)

            i = np.arange(b.size)
            j = i % a.size
            a_multiplier = sp.csr_matrix((a_multiplier, (i, j)))

            b_multiplier = sp.kron(sp.eye(b.size / a.size, b.size / a.size),
                                   sp.dia_matrix((b_multiplier, 0), (b_multiplier.size, b_multiplier.size)))

            if not isinstance(a, np.ndarray): a_x_b.add_ops(a, a_multiplier)
            if not isinstance(b, np.ndarray): a_x_b.add_ops(b, b_multiplier)

            if __DEBUG_MODE__:
                a_x_b._DEBUG_perturb = _DEBUG_perturb(self) * base(a) \
                                     + base(self) * _DEBUG_perturb(a)
                _DEBUG_check(a_x_b)
        return a_x_b

    def __rmul__(self, a):
        return self.__mul__(a)

    def __imul__(self, a):
        if isinstance(a, numbers.Number):
            self._base *= a
            self.self_ops(a)
            if __DEBUG_MODE__:
                self._DEBUG_perturb *= a
                _DEBUG_check(self)
        else:
            self._base *= a._base
            multiplier = sp.dia_matrix((np.ravel(a._base), 0), (a.size,a.size))
            self.self_ops(multiplier)
            multiplier = sp.dia_matrix((np.ravel(self._base), 0), (self.size,self.size))
            self.add_ops(a, multiplier)
            if __DEBUG_MODE__:
                self._DEBUG_perturb = _DEBUG_perturb(self) * base(a) \
                                    + base(self) * _DEBUG_perturb(a)
                _DEBUG_check(self)
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
        multiplier = sp.dia_matrix((a * np.ravel(self._base)**(a-1), 0), (self.size,self.size))
        self_to_a.add_ops(self, multiplier)
        if __DEBUG_MODE__:
            self_to_a._DEBUG_perturb = a * self._base**(a-1) \
                                     * _DEBUG_perturb(self)
            _DEBUG_check(self_to_a)
        return self_to_a
    
    # ------------------ algebraic function ----------------- #

    def exp(self):
        return exp(self)

    # ------------------ indexing ----------------- #

    def __getitem__(self, ind):
        self_i = adarray(self._base[ind])

        j = np.ravel(self._ind[ind])
        i = np.arange(j.size)
        multiplier = sp.csr_matrix((np.ones(j.size), (i,j)),
                                   shape=(j.size, self.size))
        self_i.add_ops(self, multiplier)

        if __DEBUG_MODE__:
            self_i._DEBUG_perturb = _DEBUG_perturb(self)[ind]
            _DEBUG_check(self_i)
        return self_i

    def __setitem__(self, ind, a):
        data = np.ones(self.size)
        data[self._ind[ind]] = 0
        multiplier = sp.dia_matrix((data, 0), (data.size, data.size))
        self.self_ops(multiplier)

        self._base.__setitem__(ind, base(a))

        if hasattr(a, '_base'):
            i = np.ravel(self._ind[ind])
            j = np.arange(i.size)
            multiplier = sp.csr_matrix((np.ones(j.size), (i,j)),
                                       shape=(self.size, a.size))
            self.add_ops(a, multiplier)

        if __DEBUG_MODE__:
            self._DEBUG_perturb[ind] = _DEBUG_perturb(a)
            _DEBUG_check(self)

    # ------------------ str, repr ------------------ #

    def __str__(self):
        return str(self._base)

    def __repr__(self):
        return 'ad' + repr(self._base)

    # ------------------ differentiation ------------------ #

    def diff(self, u):
        return self._diff_recurse(u, self.i_ops())
    
    def _diff_recurse(self, u, i_f_ops):
        def multiply_ops(op0, op1):
            if op0 is 0 or op1 is 0:
                return 0
            else:
                return op0 * op1

        if i_f_ops == 0:  # I got to the bottom
            if u is self:
                return sp.eye(u.size, u.size)
            else:
                # return sp.csr_matrix((self.size, u.size))
                return 0
        else:
            op = self._ops[i_f_ops - 1]
            if len(op) == 1:  # self operation
                multiplier = op[0]
                multiplier1 = self._diff_recurse(u, i_f_ops - 1)
                return multiply_ops(multiplier, multiplier1)
            else:
                other, i_other_ops, multiplier = op
                multiplier1 = other._diff_recurse(u, i_other_ops)
                other_diff = multiply_ops(multiplier, multiplier1)
                this_diff = self._diff_recurse(u, i_f_ops - 1)
                if this_diff is 0:
                    return other_diff
                elif other_diff is 0:
                    return this_diff
                else:
                    return this_diff + other_diff


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
        self.assertEqual(0, (c.diff(a) - 5 * sp.dia_matrix((b._base, 0), (N,N))).nnz)
        self.assertEqual(0, (c.diff(b) - 5 * sp.dia_matrix((a._base, 0), (N,N))).nnz)

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

        lapl = -2 * sp.eye(N-1,N-1) + sp.dia_matrix((np.ones(N-1), 1), (N-1,N-1)) \
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

        lapl_i = -2 * sp.eye(N-1,N-1) + sp.dia_matrix((np.ones(N-1), 1), (N-1,N-1)) \
                                  + sp.dia_matrix((np.ones(N-1), -1), (N-1,N-1))
        lapl_j = -2 * sp.eye(M-1,M-1) + sp.dia_matrix((np.ones(M-1), 1), (M-1,M-1)) \
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

        self.assertEqual((dRdf - sp.eye((N-1) * (M-1) * (L-1), (N-1) * (M-1) * (L-1))).nnz, 0)

        t0 = time.clock()
        dRdu = res.diff(u)
        print(time.clock() - t0)

        lapl_i = -2 * sp.eye(N-1, N-1) + sp.dia_matrix((np.ones(N-1), 1), (N-1,N-1)) \
                                  + sp.dia_matrix((np.ones(N-1), -1), (N-1,N-1))
        lapl_j = -2 * sp.eye(M-1, M-1) + sp.dia_matrix((np.ones(M-1), 1), (M-1,M-1)) \
                                  + sp.dia_matrix((np.ones(M-1), -1), (M-1,M-1))
        lapl_k = -2 * sp.eye(L-1, L-1) + sp.dia_matrix((np.ones(L-1), 1), (L-1,L-1)) \
                                  + sp.dia_matrix((np.ones(L-1), -1), (L-1,L-1))
        lapl = sp.kron(sp.kron(lapl_i, sp.eye(M-1, M-1)), sp.eye(L-1, L-1)) / dx**2 \
             + sp.kron(sp.kron(sp.eye(N-1, N-1), lapl_j), sp.eye(L-1, L-1)) / dy**2 \
             + sp.kron(sp.kron(sp.eye(N-1, N-1), sp.eye(M-1, M-1)), lapl_k) / dz**2
        self.assertEqual((dRdu - lapl).nnz, 0)

        # pylab.figure()
        # pylab.spy(lapl, marker='.')


class _Burgers1dTest(unittest.TestCase):
    def firstOrderFlux(self, u):
        u = hstack([0, u, 0])
        f = u**2 / 2
        f_max = maximum(f[1:], f[:-1])
        f_min = minimum(0, minimum(f[1:], f[:-1]))
        return (u[1:] <= u[:-1]) * f_max + (u[1:] > u[:-1]) * f_min

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
