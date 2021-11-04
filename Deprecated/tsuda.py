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

class SimpleModel(Dynamic):
    def __init__(self, prec):       
	self.name = "tsuda"
	self.nbranches = 2
	RI = self.field = RealIntervalField(prec=prec)
        R = self.real_field = RealField(prec=prec)
	self.A = RI(0.08)
	self.C = RI(0.1)
       	self.pi = all.arccos(RI(-1))
        self.e = all.exp(RI(1))
	self.alpha = R(safe_interval_newton(lambda x: x + (self.A)*all.sin(RI('4')*self.pi*x) + self.C, lambda x: RI(1) + RI(4)*self.A*self.pi*all.cos(RI(4)*self.pi*x), RI(0,1), RI(1), 2**(5-prec)))
        self._domain = RI('0','1')
	self.branch_intervals = [RI(0,self.alpha),RI(self.alpha,1)]
	self.nbranches = 2

    def preimage(self, x, branch, epsilon):
        RI = self.field
        if branch == 0:
            f = lambda(x): self.f1(x)
            fprime = lambda(x): self.f_prime(x)
            dom = RI(0, self.alpha)
        elif branch == 1:
            f = lambda(x): self.f2(x)
            fprime = lambda(x): self.f_prime(x)
            dom = RI(self.alpha, 1)
        else:
            raise ValueError, 'Invalid branch'
        warn_if_not_in_01(x)
        #print x, epsilon
        return safe_interval_newton(f, fprime, dom, x, epsilon)
    def f1(self, x):
	RI = self.field
	return x + self.A*(all.sin(RI(4)*self.pi*x))+self.C
    def f2(self, x):
	RI = self.field
	return x + self.A*(all.sin(RI(4)*self.pi*x))+self.C-RI(1)
    def f(self, x):
	RI = self.field
	if x.upper() <= self.alpha:
	   return self.f1(x)
        if x.lower() < self.alpha:
	   return self.f1(x).union(self.f2(x))
        return self.f2(x)
    def f_prime(self, x):
	RI = self.field
	return RI(1) + RI(4)*self.pi*self.A*all.cos(4*self.pi*x)
    def f_prime_abs_log(self, x):
	RI = self.field
	return abs(log(RI(1) + RI(4)*self.pi*self.A*all.cos(RI(4)*self.pi*x)))
    def f_second(self, x):
	RI = self.field
	return -RI(16)*self.pi^2*self.A*all.sin(RI(4)*self.pi*x)
