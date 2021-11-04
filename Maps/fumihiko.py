# -*- coding: utf-8 -*-
"""
Example, discretize on a grid and estimate, in L1 or Linfinity
"""

from __future__ import division
from sage import all
from dynamic import *
from eigenerr import *
from sparse import *
from estimator import *
from plotting import *
from interval import interval_newton
from safe_interval_newton import *

class FumihikoModel(Dynamic):
    def __init__(self, prec):
	self.name = "fumihiko"
	RI = self.field = RealIntervalField(prec=prec)
        R = self.real_field = RealField(prec=prec)

	self.c = RI(0.58)
	self.alpha = RI(0.7)
	self.epsilon = (self.alpha**3) / (RI(2)*(self.alpha**4 + self.alpha**2 + RI(1)))
        
	self.alpha_m_2einv = self.alpha - RI(1)/(RI(2)*self.epsilon)
        self.alpha_epsilon = self.alpha * self.epsilon
	self.alpha_abslog = abs(log(self.alpha))
	self.alpha_m_2einv_abslog = abs(log(self.alpha_m_2einv))
        self._domain = RI('0','1')
        self.bound1 = self.c-self.epsilon
        self.bound2 = self.c+self.epsilon
        self.branch_intervals = [(RI(0),self.bound1),
                                 (self.bound1,self.bound2),
                                 (self.bound2,RI(1))]
	self.nbranches = 3

    def preimage(self, x, branch, epsilon):
        RI = self.field
        if branch == 0:
            f = lambda(x): self.f1(x)
            fprime = lambda(x): self.alpha
            dom = RI(0).union(self.bound1) #union here
        elif branch == 1:
            f = lambda(x): self.f2(x)
            fprime = lambda(x): self.alpha_m_2einv
            dom = self.bound1.union(self.bound2) #union here
        elif branch == 2:
            f = lambda(x): self.f3(x)
            fprime = lambda(x): self.alpha
            dom = self.bound2.union(RI(1)) #union here
        else:
            raise ValueError, 'Invalid branch'
        warn_if_not_in_01(x)
        return safe_interval_newton(f, fprime, dom, x, epsilon)

    def f1(self, x):
	RI = self.field
	return self.alpha*(x - self.c) + RI(1)
    def f2(self, x):
	RI = self.field
	return self.alpha_m_2einv*(x - self.c - self.epsilon) + self.alpha_epsilon
    def f3(self, x):
	RI = self.field
	return self.alpha*(x - self.c)

    def f1_prime(self, x):
	RI = self.field
	return self.alpha
    def f2_prime(self, x):
	RI = self.field
	return self.alpha_m_2einv
    def f3_prime(self, x):
	RI = self.field
	return self.alpha

    def f_branch(self, x, branch): # will be used by the new assembler, if available
        if branch == 0:
            return self.f1(x)
        elif branch == 1:
            return self.f2(x)
        else:
            return self.f3(x)

    def f(self, x):
	RI = self.field
	if x.upper() <= self.bound1.lower():
            return self.f1(x)
        elif x.lower() < self.bound1.upper(): #intersects
	    return self.f1(x).union(self.f2(x))
        elif x.upper() <= self.bound2.lower():
	    return self.f2(x)
        elif x.lower() <= self.bound2.upper():
	    return self.f2(x).union(self.f3(x))
        else:
            return self.f3(x)

    def f_prime(self, x):
	RI = self.field
	if x.upper() <= self.bound1.lower():
	   return self.alpha
        elif x.lower() < self.bound1.upper(): #intersects
	   return self.alpha.union(self.alpha_m_2einv)
        elif x.upper() <= self.bound2.lower():
	   return self.alpha_m_2einv
        elif x.lower() <= self.bound2.upper():
	   return self.alpha_m_2einv.union(self.alpha)
        else:
            return self.alpha

    def f_prime_abs_log(self, x):
	RI = self.field
	if x.upper() <= self.bound1.lower():
	   return self.alpha_abslog
        elif x.lower() < self.bound1.upper(): #intersects
	   return self.alpha_abslog.union(self.alpha_m_2einv_abslog)
        elif x.upper() <= self.bound2.lower():
	   return self.alpha_m_2einv_abslog
        elif x.lower() <= self.bound2.upper():
	   return self.alpha_m_2einv_abslog.union(self.alpha_abslog)
        else:
            return self.alpha_abslog

    def f_second(self, x):
	RI = self.field
	return RI(0)
