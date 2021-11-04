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

class SimpleModel(Dynamic):
    def __init__(self, prec):
        self.name = "ricardo_test1"
        #self.name = "simple_model"
        
        RI = self.field = RealIntervalField(prec=prec)
        R = self.real_field = RealField(prec=prec)

        self.e = exp(RI(1))
        self._domain = RI('0','1')
        self.nbranches = 3

        self._limit12 = RI('0.3')
        self._limit23 = RI('0.6')
        
        self.branch_intervals = [ (RI('0'), RI('0.3')),
                                  (RI('0.3'), RI('0.6')),
                                  (RI('0.6'), RI('1')) ]

    def f1(self, x):
        RI = self.field
        return RI('0.95')-RI('6')*x*(RI('0.8')-x)

    def f1_prime(self, x):
        RI = self.field
        return -RI('6')*RI('0.8') + RI('12')*x

    def f1_second(self, x):
        RI = self.field
        return RI('12')

    def f2(self, x):
        RI = self.field
        return RI('0.95')+RI('6')*(x+RI('0.2'))*(x-RI('0.6'))

    def f2_prime(self, x):
        RI = self.field
        return RI('6')*(-RI('0.4')) + RI('12')*x

    def f2_second(self, x):
        RI = self.field
        return RI('12')

    def f3(self, x):
        RI = self.field
        return RI('0.95')-(x-RI('0.6'))
        #return RI('0.95')-(exp(x)-exp(RI('0.6')))

    def f3_prime(self, x):
        RI = self.field
        return RI('-1')
        #return -exp(x)

    def f3_second(self, x):
        RI = self.field
        return RI('0')
        #return -exp(x)
        
    def preimage(self, x, branch, epsilon):
        RI = self.field
        if branch == 0:
            f = lambda(x): self.f1(x)
            fprime = lambda(x): self.f1_prime(x)
            dom = RI('0', '0.3')
        elif branch == 1:
            f = lambda(x): self.f2(x)
            fprime = lambda(x): self.f2_prime(x)
            dom = RI('0.3', '0.6')
        elif branch == 2:
            f = lambda(x): self.f3(x)
            fprime = lambda(x): self.f3_prime(x)
            dom = RI('0.6', '1.0')
        else:
            raise ValueError, 'Invalid branch'
            
        warn_if_not_in_01(x)
        #print x, epsilon
        return interval_newton(f, fprime, dom, x, epsilon)

    def f(self, x):
        if x.upper() <= self._limit12:
            return self.f1(x)
        if x.lower() < self._limit12:
            return self.f1(x).union(self.f2(x))
        if x.upper() <= self._limit23:
            return self.f2(x)
        if x.lower() < self._limit23:
            return self.f2(x).union(self.f3(x))
        return self.f3(x)

    def f_prime(self, x):
        if x.upper() <= self._limit12:
            return self.f1_prime(x)
        if x.lower() < self._limit12:
            return self.f1_prime(x).union(self.f2_prime(x))
        if x.upper() <= self._limit23:
            return self.f2_prime(x)
        if x.lower() < self._limit23:
            return self.f2_prime(x).union(self.f3_prime(x))
        return self.f3_prime(x)

    def f_prime_abs_log(self, x):
        if x.upper() <= self._limit12:
            return self.f1_prime(x).abs().log()
        if x.lower() < self._limit12:
            return self.f1_prime(x).abs().log().union(self.f2_prime(x).abs().log())
        if x.upper() <= self._limit23:
            return self.f2_prime(x).abs().log()
        if x.lower() < self._limit23:
            return self.f2_prime(x).abs().log().union(self.f3_prime(x).abs().log())
        return self.f3_prime(x).abs().log()

    def f_second(self, x):
        if x.upper() <= self._limit12:
            return self.f1_second(x)
        if x.lower() < self._limit12:
            return self.f1_second(x).union(self.f2_second(x))
        if x.upper() <= self._limit23:
            return self.f2_second(x)
        if x.lower() < self._limit23:
            return self.f2_second(x).union(self.f3_second(x))
        return self.f3_second(x)
