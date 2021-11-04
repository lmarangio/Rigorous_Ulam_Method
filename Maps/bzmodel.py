# -*- coding: utf-8 -*-
"""
Example, discretize on a grid and estimate, in L1 or Linfinity
"""

from __future__ import division

from dynamic import *
from eigenerr import *
from sparse import *
from estimator import *
from sage.all import *
from plotting import *
from interval import interval_newton

class BZModel(Dynamic):
    def __init__(self, prec):
        self.name = "bzmodel"
        
        RI = self.field = RealIntervalField(prec=prec)
        R = self.real_field = RealField(prec=prec)
        self._domain = RI('0','1')
        self.nbranches = 2

        # constants necessary to define the (original) map
        self._fence = R('0.3')
        self._1q3 = RI('1.0') / RI('3.0')
        self._2q3 = RI('2.0') / RI('3.0')
        self._m2q3 = - RI('2.0') / RI('3.0')
        self._m5q3 = - RI('5.0') / RI('3.0')
        self._0d125 = RI('0.125')
        
        self._0d3 = RI('0.3')
        self._0d5 = RI('0.5')
        self._m10q3 = -RI('10')/RI('3')
        self._19 = RI('19')
        self._38 = RI('38')
        self._18 = RI('18')
        self._17 = RI('17')
        self._10 = RI('10')
        self._2 = RI('2')
        self._d1 = - (RI('0.175')**(-RI('5')/RI('3'))) * RI('2') / RI('9') \
                   - (RI('0.175')**(-RI('2')/RI('3'))) / RI('3')
        self._3 = RI('3')

        self._0 = RI('0')
        self._1 = RI('1')
        self._m1 = RI('-1')
        self._pm1 = RI('-1','1')

        

        # a = 0.50607357..., computed imposing that f1's derivative is 0 for x = 0.3
        dx = self._0d3 - self._0d125
        self._d50607357 = - (dx ** self._1q3) + self._1q3 * (dx ** self._m2q3)

        # c = 0.121205692..., computed imposing that the two functions glue well
        self._b = RI('0') # set temporarily to 0
        self._d121205692 = RI('1') # set temporarily to 1 to perform computation below
        self._d121205692 = self.orig_f1(RI('0.3')) / self.orig_f2(RI('0.3'))

        # computation of b:
        # b is the parameter such that T^5(0.3) (that is T^4 of the critical VALUE) is
        # an expanding fixed point. We do the following:
        # 1. b is temporarily set to 0 in this class, so T_b(x) can be computed as T(x)+b
        # 2. we define functions of b that are T_b^5(0.3)-T_b^6(0.3) and its derivative
        # 3. the derivative is obtained composing 2nd order functions
        # 4. we apply the interval Newton algorithm starting from the interval
        #    RI('0.02326', '0.0233'), since there is a strong expansion starting from a
        #    bigger interval can be problematic (or we could use the safe version that
        #    begins with a few bisection steps).
        def tmp_f(b):
            f1 = self.orig_f1
            f2 = self.orig_f2
            pf = f1(f1(f1(f2(f1(self._fence)+b)+b)+b)+b)+b
            return pf - (f2(pf) + b)

        def tmp_f_prime(b):
            # let's use 2nd order versions of f1/f2, pair is: (value, derivative in b)
            f1o2 = lambda jet: (self.orig_f1(jet[0])+b, jet[1] * self.orig_f1_prime(jet[0]) + RI(1))
            f2o2 = lambda jet: (self.orig_f2(jet[0])+b, jet[1] * self.orig_f2_prime(jet[0]) + RI(1))
            jet_pf = f1o2(f1o2(f1o2(f2o2(f1o2((self._fence, RI(0)))))))
            jet_Tpf = f2o2(jet_pf)
            return jet_pf[1]-jet_Tpf[1]

        self._b = interval_newton(tmp_f, tmp_f_prime, RI('0.02326', '0.0233'), RI(0), 2**(10-prec))
        #print('b:', self._b.endpoints())
        #print('diam:', self._b.diameter())
        #print('dist to old:', (self._b-RI('0.0232885279')).abs().upper())

        # OUT_OF_DATE # what is the exact value of this parameter? Japanese guys...?
        # OUT_OF_DATE # self._b = RI('0.0232885279')

        # A scaling so that [0,1] is mapped into [0.01, 0.99],
        # magnifying as much as possible. Notice that these
        # constants depend on the parameter b, for different b
        # compute new constants as well!
        self._start = RI('0.014')
        self._scale = RI('0.817')
        self._iscale = RI('1.0') / self._scale

        # to allow the columnwise computation of the ulam matrix
        self._sing1 = (self._0d125 - self._start) * self._iscale
        self._real_fence = (self._fence - self._start) * self._iscale
        self._dom1 = RI(self._real_fence).union(RI('0'))
        self._dom2 = RI(self._real_fence).union(RI('1'))
        self.branch_intervals = [(RI(0), self._real_fence),
                                 (self._real_fence, RI('1'))]

    def sign(self, x):
        if(x.lower() >= self._0):
            return self._1
        elif(x.upper() <= self._0):
            return self._m1
        else:
            return self._pm1

    def thrdroot(self, x):
        return self.sign(x) * (x.abs() ** self._1q3)

    def orig_f1(self, x):
        return (self.thrdroot(x-self._0d125) + self._d50607357) * exp(-x) + self._b

    # when does it get negative? let's check:
    # Rx.<x> = PolynomialRing(RR)
    # p = x^3 + BZModel(512)._d50607357.center()*x^2 - 1/3
    # [(r[0]**3 + 0.125, r[1]) for r in p.roots()]
    def orig_f1_prime(self, x):
        dx = x - self._0d125
        retv = ((self._1q3/dx - self._1) * self.thrdroot(dx) - self._d50607357) * exp(-x)

        # HACK: never return something containing 0, or the interval Newton will fail.
        # this is because (sigh) the dividing by the interval [0,x] we always get
        # [-infinity, +infinity], no way to make specify "+0" to get [0, +infinity]
        eps = self.real_field.epsilon()
        return self.field(max(eps, retv.lower()), max(eps, retv.upper()))

    def orig_f1_second(self, x):
        dx = x-self._0d125
        val = self._1q3/dx
        return ((self._1 - self._2*(val + val.square())) * self.thrdroot(dx) +
                self._d50607357) * exp(-x)

    def orig_f2(self, x):
        return self._d121205692 * ((10 * x * exp(self._m10q3 * x)) ** self._19) + self._b

    def orig_f2_prime(self, x):
        xp = self._10 * exp(self._m10q3 * x)
        retv = self._d121205692 * self._19 * ((x * xp) ** self._18) * \
                                        xp * (self._1 + x * self._m10q3)
        # HACK: never return something containing 0, or the interval newton will fail.
        # this is because (sigh) the dividing by the interval [0,x] we always get
        # [-infinity, +infinity], no way to make specify "+0" to get [0, +infinity]
        eps = self.real_field.epsilon()
        return self.field(min(-eps, retv.lower()), min(-eps, retv.upper()))

    def orig_f2_second(self, x):
        xp = self._10 * exp(self._m10q3 * x)
        return self._d121205692 * self._19 * ((x * xp) ** self._17) * xp * xp * \
            (self._18 + self._38 * self._m10q3*x + self._19*self._m10q3*self._m10q3*x*x)

    # estimates rigorously the integral of log|f'| in the interval [p, q],
    # for 0.125-2^-6 < p < 0.125 < q < 0.125+2^-6.
    #
    # It returns the rigorous estimate of the integral,
    # that is a POSITIVE interval value.
    def orig_bound_int_p_q_log_f1_prime(self, p, q):
        eps_p = self._0d125 - p
        eps_q = q - self._0d125
        ubound =  self._m2q3 * (eps_p * (eps_p.log() - self._1) + \
                                eps_q * (eps_q.log() - self._1)) +\
                                - (q-p)*self._3.log() - self._0d5 * (q*q - p*p)
        delta = (self.field('4').log() - self.field('5').log()) * (q-p)
        return ubound + delta.union(self._0)

    def bound_int_p_q_log_f1_prime(self, p, q):
        return self.orig_bound_int_p_q_log_f1_prime( \
             self._start + p * self._scale, \
             self._start + q * self._scale) * self._iscale
    
    def orig_bound_int_p_sing1_log_f1_prime(self, p):
        eps_p = self._0d125 - p
        q = self._0d125
        ubound =  self._m2q3 * (eps_p * (eps_p.log() - self._1)) +\
                                - (q-p)*self._3.log() - self._0d5 * (q*q - p*p)
        delta = (self.field('4').log() - self.field('5').log()) * (q-p)
        return ubound + delta.union(self._0)

    def bound_int_p_sing1_log_f1_prime(self, p):
        return self.orig_bound_int_p_sing1_log_f1_prime( \
             self._start + p * self._scale) * self._iscale

    def orig_bound_int_sing1_q_log_f1_prime(self, q):
        p = self._0d125
        eps_q = q - self._0d125
        ubound =  self._m2q3 * (eps_q * (eps_q.log() - self._1)) +\
                                - (q-p)*self._3.log() - self._0d5 * (q*q - p*p)
        delta = (self.field('4').log() - self.field('5').log()) * (q-p)
        return ubound + delta.union(self._0)

    def bound_int_sing1_q_log_f1_prime(self, q):
        return self.orig_bound_int_sing1_q_log_f1_prime( \
             self._start + q * self._scale) * self._iscale
    
    # estimates rigorously the integral of log|f'| in the interval [x, 0.3]
    #
    # It returns the rigorous estimate of the integral,
    # that is a NEGATIVE interval value.
    def orig_bound_int_p_0d3_log_f1_prime(self, p):
        RI = self.field
        epsilon = self._0d3 - p
        # compute d2 = -phi(p)/(0.3 - p), to have an upper bound
        dp = p - self._0d125
        d2 = (+ dp ** self._1q3 + self._d50607357 \
                 - self._1q3 * (dp ** self._m2q3)) / epsilon
        d = -d2.union(self._d1)
        return (d.log() + epsilon.log()-1) * epsilon \
                  - self._0d5*(self._0d3*self._0d3 - p*p)

    def bound_int_p_fence_log_f1_prime(self, x):
        return self.orig_bound_int_p_0d3_log_f1_prime( \
                self._start + x * self._scale) * self._iscale

    # estimates (actually computes exactly) the integral of log|f'|
    # in the interval [0.3, x], it is valid estimates of the L^1 norm
    # for x < 0.303 (because log|f'| is invariably negative in the
    # interval [0.3, 0.303]).
    #
    # It returns the rigorous estimate of the integral,
    # that is a NEGATIVE interval value.
    def orig_bound_int_0d3_q_log_f2_prime(self, q):
        epsilon = q - self._0d3
        return (log(self._d121205692)       \
            + log(self._19)                 \
            + self._19 * log(self._10)      \
            - self._38                      \
            - log(self._0d3)) * epsilon     \
            + 18 * (q * log(q) - self._0d3 * log(self._0d3))   \
            + epsilon * log(epsilon)                           \
            + self._m10q3 * self._19 * self._0d5 * epsilon**2  \

    def bound_int_fence_q_log_f2_prime(self, q):
        return self.orig_bound_int_0d3_q_log_f2_prime(   \
                self._start + q * self._scale) * self._iscale

    # just scale by a small factor to make everything fit into [0,1]
    def f1(self, x):
        return self._iscale * (self.orig_f1(self._start + x * self._scale) - self._start)

    # just scale by a small factor to make everything fit into [0,1]
    def f1_prime(self, x):
        return self.orig_f1_prime(self._start + x * self._scale)

    def f1_second(self, x):
        return self._scale * self.orig_f1_second(self._start + x * self._scale)

    # just scale by a small factor to make everything fit into [0,1]
    def f2(self, x):
        return self._iscale * (self.orig_f2(self._start + x * self._scale) - self._start)

    # just scale by a small factor to make everything fit into [0,1]
    def f2_prime(self, x):
        return self.orig_f2_prime(self._start + x * self._scale)

    # just scale by a small factor to make everything fit into [0,1]
    def f2_second(self, x):
        return self._scale * self.orig_f2_second(self._start + x * self._scale)

    def preimage(self, x, branch, epsilon):
        if branch == 0:
            f = lambda(x): self.f1(x)
            fprime = lambda(x): self.f1_prime(x)
            dom = self._dom1
        elif branch == 1:
            f = lambda(x): self.f2(x)
            fprime = lambda(x): self.f2_prime(x)
            dom = self._dom2
        else:
            raise ValueError, 'Invalid branch'
            
        warn_if_not_in_01(x)
        return interval_newton(f, fprime, dom, x, epsilon)

    def orig_f(self, x):
        if x.upper() <= self._fence:
            return self.orig_f1(x)
        if x.lower() >= self._fence:
            return self.orig_f2(x)
        return self.orig_f1(x).union(self.orig_f2(x))

    def orig_f_prime(self, x):
        if x.upper() <= self._fence:
            return self.orig_f1_prime(x)
        if x.lower() >= self._fence:
            return self.orig_f2_prime(x)
        return self.orig_f1_prime(x).union(self.orig_f2_prime(x))

    def orig_f_second(self, x):
        if x.upper() <= self._fence:
            return self.orig_f1_second(x)
        if x.lower() >= self._fence:
            return self.orig_f2_second(x)
        return self.orig_f1_second(x).union(self.orig_f2_second(x))

    # just scale by a small factor to make everything fit into [0,1]
    def f(self, x):
        return self._iscale * (self.orig_f(self._start + x * self._scale) - self._start)

    # just scale by a small factor to make everything fit into [0,1]
    def f_prime(self, x):
        return self.orig_f_prime(self._start + x * self._scale)

    # just scale by a small factor to make everything fit into [0,1]
    def f_second(self, x):
        return self._scale * self.orig_f_second(self._start + x * self._scale)
