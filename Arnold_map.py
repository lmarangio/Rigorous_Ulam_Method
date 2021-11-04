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
		self.name = "Arnold_map"
		self.nbranches = 6
		RI = self.field = RealIntervalField(prec=prec)
        	R = self.real_field = RealField(prec=prec)
		self.K = RI('1.4')
		self.omega = RI('0.716')
       		self.pi = RI(all.pi) 
        	self.e = RI(all.e) 
        	func = lambda x: x - ((self.K) / (RI('2')*self.pi))*all.sin(RI('2')*self.pi*x) + self.omega
        	deriv = lambda x: RI(1) - self.K*all.cos(RI(2)*self.pi*x)
		sec_deriv = lambda x: RI(2)*self.K*self.pi*all.sin(RI(2)*self.pi*x)
		self.alpha = RI(safe_interval_newton(func, deriv, RI('0','1'), RI('1'), 2**(5-prec))) #INTERVAL!
        	self._domain = RI('0','1')
		self._d1=all.arccos(RI('1')/self.K)/(RI('2')*self.pi)
		self._d2=RI('1')-self._d1
		self._b1=(self.alpha - self._d1)/(RI('2'))
		self._b2=(self._d2 - self.alpha)/(RI('2'))
        	self.branch_intervals = [(RI('0'), self._d1), (self._d1, RI('0.5')), (RI(0.5), self.alpha), (self.alpha, RI('0.75')), (RI('0.75'), self._d2), (self._d2, RI('1'))]
		self.nbranches = 6
	def preimage(self, x, branch, epsilon):
		RI = self.field
		if branch == 0:
			f = lambda(x): self.f1(x)
			fprime = lambda(x): self.f_prime(x)
			dom = RI('0').union(self._d1) #union here
        	elif branch == 1:
			f = lambda(x): self.f1(x)
			fprime = lambda(x): self.f_prime(x)
			dom = (self._d1).union(RI('0.5')) #union here
        	elif branch == 2:
			f = lambda(x): self.f1(x)
			fprime = lambda(x): self.f_prime(x)
			dom = (RI('0.5')).union(self.alpha) #union here
		elif branch == 3:
			f = lambda(x): self.f2(x)
			fprime = lambda(x): self.f_prime(x)
			dom = (self.alpha).union(RI('0.75')) #union here
        	elif branch == 4:
			f = lambda(x): self.f2(x)
			fprime = lambda(x): self.f_prime(x)
			dom = (RI('0.75')).union(self._d2) #union here
		elif branch == 5:
			f = lambda(x): self.f2(x)
			fprime = lambda(x): self.f_prime(x)
			dom = (self._d2).union(RI('1')) #union here
		else:
			raise ValueError, 'Invalid branch'
			warn_if_not_in_01(x)
		return	safe_interval_newton(f, fprime, dom, x, epsilon)
 	
	def f_branch(self, x, branch): # will be used by the new assembler, if available
        	
		if branch == 0:
			return min(self.field(1), self.f1(x))
		elif branch == 1:
			return min(self.field(1), self.f1(x))
		elif branch == 2:
            		return min(self.field(1), self.f1(x))
        	elif branch == 3:
            		return max(self.field(0), self.f2(x))
		else:
			return self.f2(x)
		
	def f1(self, x):
		RI = self.field
		return  x - ((self.K) / (RI('2')*self.pi))*all.sin(RI('2')*self.pi*x) + self.omega
	def f2(self, x):
		RI = self.field
		return  x - ((self.K) / (RI('2')*self.pi))*all.sin(RI('2')*self.pi*x) + self.omega - RI('1')
	def f(self, x):
		RI = self.field
		if x.upper() <= self.alpha:
			return self.f1(x)
        	elif x.lower() < self.alpha:
			return self.f1(x).union(self.f2(x))
		return self.f2(x)
	def f_prime(self, x):
		RI = self.field
		return RI('1') - self.K*all.cos(RI('2')*self.pi*x)
	def f_lift(self, x):
		RI = self.field
		return self.omega - ((self.K) / (RI('2')*self.pi))*all.sin(RI('2')*self.pi*x)
	def f_prime_log_abs(self, x):
		RI = self.field
		return all.log(abs(RI('1') - self.K*all.cos(RI('2')*self.pi*x)))
	def f_second(self, x):
		RI = self.field
		return RI('2')*self.K*self.pi*all.sin(RI('2')*self.pi*x)


