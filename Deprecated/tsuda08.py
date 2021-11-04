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
	self.name = "tsuda08"
	self.nbranches = 6
	RI = self.field = RealIntervalField(prec=prec)
        R = self.real_field = RealField(prec=prec)

        # Mau's note: as surprising as it looks, with an A as small as
        # 0.08 the derivative becomes negative!!
        # indeed:
        # sage: RR(4*pi*0.08)
        #   1.00530964914873
        # If the derivative becomes negative somewhere you need to split [0,1]
        # in intervals where the function is monotomic.
        # With A=0.07 everything is OK.
	self.A = RI(0.08)
	self.C = RI(0.1)
       	self.pi = RI(all.pi) #all.arccos(RI(-1)) # Mau's note: C'mon!
        self.e = RI(all.e) #all.exp(RI(1))
        func = lambda x: x + (self.A)*all.sin(RI('4')*self.pi*x) + self.C
        deriv = lambda x: RI(1) + RI(4)*self.A*self.pi*all.cos(RI(4)*self.pi*x)
	sec_deriv = lambda x: -RI(16)*self.A*self.pi^2*all.sin(RI(4)*self.pi*x)
	self.alpha = RI(safe_interval_newton(func, deriv, RI(0,1), RI(1), 2**(5-prec))) #INTERVAL!
        self._domain = RI('0','1')
	self.deca = all.arccos(-RI(1)/(RI(4)*self.A*self.pi))/(RI(4)*self.pi)
	self.decb = RI(0.5) - self.deca
	self.decc = RI(0.5) + self.deca
	self.decd = RI(1) - self.deca
        # Mau's note: branch_intervals is attumed to be an array of PAIRs of intervals
	#self.branch_intervals = [RI(0,self.alpha), RI(self.alpha,1)] #wrong! ;)
        self.branch_intervals = [(RI(0),self.deca), (self.deca,self.decb), (self.decb,self.decc), (self.decc,self.decd), (self.decd,self.alpha), (self.alpha, RI(1))]
	self.nbranches = 6

    def preimage(self, x, branch, epsilon):
        RI = self.field
        if branch == 0:
            f = lambda(x): self.f1(x)
            fprime = lambda(x): self.f_prime(x)
            dom = RI(0).union(self.deca) #union here
        elif branch == 1:
            f = lambda(x): self.f1(x)
            fprime = lambda(x): self.f_prime(x)
            dom = (self.deca).union(self.decb) #union here
	elif branch == 2:
	    f = lambda(x): self.f1(x)
            fprime = lambda(x): self.f_prime(x)
            dom = (self.decb).union(self.decc) #union here
	elif branch == 3:
	    f = lambda(x): self.f1(x)
            fprime = lambda(x): self.f_prime(x)
            dom = (self.decc).union(self.decd) #union here
	elif branch == 4:
	    f = lambda(x): self.f1(x)
            fprime = lambda(x): self.f_prime(x)
            dom = (self.decd).union(self.alpha) #union here
	elif branch == 5:
	    f = lambda(x): self.f2(x)
            fprime = lambda(x): self.f_prime(x)
            dom = (self.alpha).union(RI(1)) #union here
	else:
            raise ValueError, 'Invalid branch'
        warn_if_not_in_01(x)
        #print x, branch
        return safe_interval_newton(f, fprime, dom, x, epsilon)
    def f1(self, x):
	RI = self.field
	return x + self.A*(all.sin(RI(4)*self.pi*x)) + self.C
    def f2(self, x):
	RI = self.field
	return x + self.A*(all.sin(RI(4)*self.pi*x)) + self.C - RI(1)
    def f_branch(self, x, branch): # will be used by the new assembler, if available
        if branch == 4:
            return min(self.field(1), self.f1(x))
        elif branch == 5:
            return max(self.field(0), self.f2(x))
	else:
	    return self.f1(x)
    def f(self, x):
	RI = self.field
	if x.upper() <= self.alpha:
	   return self.f1(x)
        elif x.lower() < self.alpha:
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
	return -RI(16)*self.pi*self.pi*self.A*all.sin(RI(4)*self.pi*x)
