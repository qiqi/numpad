from __future__ import division
from math import factorial
from pylab import *
from numpy import *

_unzip = lambda zipped: zip(*zipped) # unzip a list of tuples

def _C(n, k):
    # binomial coefficient == n! / (i!(n - i)!)
    return factorial(n) / (factorial(k) * factorial(n - k))


class Bezier(object):

    def __init__(self, P):
        """
        construct bezier curve

        P == list of control points
        """
        self.X, self.Y = _unzip(P)
        self._n = range(len(P)) # control point iterator

    def __call__(self, t):
        """
        domain t in [0, 1]

        return point on bezier curve at t
        """
        assert 0 <= t <= 1 # t in [0, 1]
        X, Y, _n = self.X, self.Y, self._n
        x, y, n = 0, 0, _n[-1] # initial x, y return values and n
        for i in _n:
            b_i = _C(i, n) * t**i * (1 - t)**(n - i) # bernstein polynomial
            # mult ith control point by ith bernsteim polynomial
            # t = 0 maps to first control point
            # t = 1 maps to nth control point
            x += X[i] * b_i
            y += Y[i] * b_i
        return x, y


class Bspline(object):

    def __init__(self, P, t, k = None):
        """
        construct Bspline object
        uses Cox-DeBoor

        P == vector of two-dimensional control points
        t == vector of non-decreasing real numbers
        k == degree of curve

        identities:
        P = (P[0], ... P[n]); n = len(P) - 1
        t = (t[0], ... t[m]); m = len(t) - 1
        k = m - n - 1
        m = n + k + 1
        n = m - k - 1
        """
        m, n = len(t) - 1, len(P) - 1
        if not k: k = m - n - 1
        else: assert m == n + k + 1
        self.k, self.t = k, t
        self.X, self.Y = _unzip(P) # points in X, Y components
        self._deboor() # evaluate

    def __call__(self, t_):
        """
        S(t) = sum(b[i][k](t) * P[i] for i in range(0, n))
        domain: t in [t[k - 1], t[n + 1]]

        returns point on Bspline at t_
        """
        k, t = self.k, self.t
        m = len(t) - 1
        n = m - k - 1
        # t in [t[k - 1], t[n + 1]]
        assert all(t[k - 1] <= t_)
        assert all(t_ <= t[n + 1])
        X, Y, b = self.X, self.Y, self.b
        x, y, _n = 0, 0, range(n + 1) # initial return values, iterator over P
        for i in _n:
            b_i = b[i][k](t_)
            x += X[i] * b_i
            y += Y[i] * b_i
        return x, y

    def _deboor(self):
        # de Boor recursive algorithm
        # S(t) = sum(b[i][k](t) * P[i] for i in range(0, n))
        #
        # b[i][k] = {
        #     if k == 0:
        #         t[i] <= t_ < t[i+1]
        #     else:
        #         a[i][k](t)*b[i][k-1](t)+(1-a[i+1][k](t))*b[i+1][k-1](t)
        # }
        #
        # a[i][k] = {
        #     if t[i] == t[i+k]:
        #         0
        #     else:
        #         (t_-t[i])/(t[i+k]-t[i])
        # }
        #
        # NOTE: for b[i][k](t), must iterate to t[:-1];
        # the number of [i, i + 1) spans in t
        k, t = self.k, self.t
        m = len(t) - 1 # iterate to t[:-1]
        a, b, _k_, _m_ = [], [], range(k + 1), range(m)
        for i in _m_:
            a.append([]); b.append([]) # a[i]; b[i]
            for k in _k_:
                a[i].append(None) # a[i][k]
                # if k == 0: b[i][k](t) is a step function in [t[i], t[i + 1])
                if k == 0:
                    def bi(t_, i=i):
                        return array(logical_and(t[i] <= t_, t[i+1] > t_))
                    b[i].append(bi)
                # if m < i + k: b[i][k](t) undefined
                elif m < i + k:
                    b[i].append(lambda t_: False)
                # else: calculate b[i][k](t)
                else:
                    # if t[i] == t[i + k]: a[i][k] undefined
                    if t[i] == t[i + k]:
                        a[i][k] = lambda t_: False
                    # else: calculate a[i][k](t)
                    else:
                        # a[i][k](t) = (t_ - t[i]) / (t[i + k] - t[i])
                        a[i][k] = lambda t_, i=i, k=k: ((t_ - t[i]) /
                                                        (t[i + k] - t[i]))
                    # b[i][k](t) = a[i][k](t) * b[i][k - 1](t) +
                    #              (1 - a[i + 1][k](t)) * b[i + 1][k - 1](t)
                    b[i].append(lambda t_, i=i, k=k:
                                a[i][k](t_) * b[i][k - 1](t_) +
                                (1 - a[i + 1][k](t_)) * b[i + 1][k - 1](t_))
        self.b = b

    def insert(self, t_):
        """
        Q[i] = (1 - a[i][k]) * P[i] + a[i][k] * P[i]
        domain: t in (t[0], t[m])

        insert new control point at t_
        """
        t = self.t
        assert t[0] < t_ < t[-1] # t_ in (t[0], t[m])
        X, Y, k = self.X, self.Y, self.k
        m = len(t) - 1
        _t_ = range(m + 1)
        # find the span containing t_
        for i in _t_:
            if t[i] <= t_ < t[i + 1]: break
        assert not i < k + 1 and not i > m - k + 1 # i not in clamp
        Q_x, Q_y = [], [] # new control points
        # iterate over replaced control points
        # set new control points
        for j in range(i - k + 1, i + 1):
            a_j = (t_ - t[j]) / (t[j + k] - t[j])
            Q_x.append((1 - a_j) * X[j - 1] + a_j * X[j])
            Q_y.append((1 - a_j) * Y[j - 1] + a_j * Y[j])
        Q_x, Q_y = tuple(Q_x), tuple(Q_y)
        self.t = t[:i + 1] + [t_] + t[i + 1:]
        self.X = X[:i - k + 1] + Q_x + X[i:]
        self.Y = Y[:i - k + 1] + Q_y + Y[i:]
        self._deboor() # re-evaluate



P = [(0, 100), (100, 0), (200, 0), (300, 100), (400, 200), (500, 200),
     (600, 100), (400, 400), (700, 50), (800, 200)]
P = transpose([arange(10), random.rand(10)])
n = len(P) - 1 # n = len(P) - 1; (P[0], ... P[n])
k = 3          # degree of curve
m = n + k + 1  # property of b-splines: m = n + k + 1
_t = 1 / (m - k * 2) # t between clamped ends will be evenly spaced
# clamp ends and get the t between them
t = k * [0] + [t_ * _t for t_ in range(m - (k * 2) + 1)] + [1] * k
print(t)

S = Bspline(P, t, k)
# insert a knot (just to demonstrate the algorithm is working)
S.insert(0.9)

t = linspace(0, 1, 100)
plot(t, S(t)[0])
plot(t, S(t)[1])
