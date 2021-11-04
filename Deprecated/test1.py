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
from interval import interval_newton
from safe_interval_newton import *
from noise_measure_computation import*


def estimate_sing_integral(z, a, fsecond):
    RI = z.parent()
    fsecond_abs = abs(fsecond)
    s = RI(fsecond_abs.lower())
    S = RI(fsecond_abs.upper())

    # flip a to the right, to simplify, in any case
    # the computation would be symmetrical
    a = max(a, 2*z-a)
    lim = min(z + ~sqrt(s*S), a) #the limit dividing the two parts

    part1 = -(lim-z)*(log(s*(lim-z)) - 1)
    part2 = (a-z)*(log(S*(a-z)) - 1) - (lim-z)*(log(S*(lim-z)) - 1)
    return (part1+part2).union(0)


class Test1(Dynamic):
    def __init__(self, prec, A = '0.12', omega = '0.12'):
        self.name = "test1"
        
        RI = self.field = RealIntervalField(prec=prec)
        R = self.real_field = RealField(prec=prec)

        self.A = RI(A)
        self.omega = RI(omega)
      
        eps = RI(2).center()**(5-self.field.prec())
        self.sep = safe_interval_newton(self.f_plain, self.f_prime,
                                         RI(0, 1), RI(1), eps)

        self._domain = RI('0','1')
        self.nbranches = 2
        
        self.branch_intervals = [ (RI('0'), self.sep),
                                  (self.sep, RI('1')) ]
        self.branch_enclosures = [ self.sep.union(RI('0')),
                                   self.sep.union(RI('1')) ]
        self.branch_increasing = [True, True]

    def f(self, x):
        RI = self.field
        if x.upper() <= self.sep.lower():
		return self.f_plain(x)
        if x.lower() >= self.sep.upper():
		return self.f_2(x)
	return RI(0,1)
	#return self.f_plain(x).union(self.f_2(x))

    def f_plain(self, x):
	RI=self.field
	return x+self.A*sin(RI(4)*RI.pi()*x)+self.omega

    def f_2(self, x):
	RI=self.field
	return self.f_plain(x)-RI(1)

    def f_prime(self, x):
        RI = self.field
        return RI(1)+self.A*RI(4)*RI.pi()*cos(RI(4)*RI.pi()*x)

    def f_second(self, x):
        RI = self.field
        return -self.A*RI(4)*RI.pi()*RI(4)*RI.pi()*sin(RI(4)*RI.pi()*x)
        
    def preimage(self, x, branch, epsilon):
        RI = self.field
        f = self.f

        if self.branch_increasing[branch]:
          fprime = lambda(x) : RI(0).max(self.f_prime(x))
        else:
          fprime = lambda(x) : RI(0).min(self.f_prime(x))
        dom = self.branch_enclosures[branch]
            
        warn_if_not_in_01(x)
        return interval_newton(f, fprime, dom, x, epsilon)
