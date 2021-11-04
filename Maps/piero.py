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

def sq(x):
    return x*x

def cb(x):
    return x*x*x

class PieroModel(Dynamic):
    def __init__(self, prec):
	self.name = "piero"
	RI = self.field = RealIntervalField(prec=prec)
        R = self.real_field = RealField(prec=prec)

        self.alpha = RI(1.64)
        self.gamma = RI(40)
        self.omega = RI(0.02)
        self.Sigma = RI(0.0001)

        self.mid = bisect_find(self.f_prime, RI(0.1,0.9), RI(0), 2**(4-prec/4))
        print('mid', self.mid)
        self.branch_intervals = [(RI(0),self.mid),
                                 (self.mid,RI(1))]
	self.nbranches = 2

    def preimage(self, x, branch, epsilon):
        RI = self.field
        f = lambda(x): self.f(x)
        if branch == 0:
            fprime = lambda(x): min(self.f_prime(x), RI(0))
            dom = RI(0).union(self.mid) #union here
        elif branch == 1:
            fprime = lambda(x): max(self.f_prime(x), RI(0))
            dom = self.mid.union(RI(1)) #union here
        else:
            raise ValueError, 'Invalid branch'
        warn_if_not_in_01(x)
        epsilon = 2**-500
        return safe_interval_newton(f, fprime, dom, x, epsilon)

    def f_branch(self, x, branch): # will be used by the new assembler, if available
        return self.f(x)

    def f(self, x):
	RI = self.field
        w,a,g,S = self.omega, self.alpha, self.gamma, self.Sigma
        sqrterm = max(RI(0), w / sq(a*(g+1)*x) + ((1-w)*(1+2*(x-1/(g+1))/(1-x))*S/(1-sq(((g-1)*x-1)/g))))
        return max(RI(0), RI(1) / (self.alpha*(self.gamma+1)*sqrt(sqrterm)))

    def f_prime(self, x):
	RI = self.field
        w,a,g,S = self.omega, self.alpha, self.gamma, self.Sigma
        sqrterm = max(RI(0), w / sq(a*(g+1)*x) + ((1-w)*(1+2*(x-1/(g+1))/(1-x))*S/(1-sq(((g-1)*x-1)/g))))
        #print([x,sqrterm])
        sqterm_der = -RI(2)*w/ (sq(a*(g+1)*x)*x) + RI(2)*(
            sq(g)*(1+g)*S*(w-1)*(x-1 + (g*(x-1)-x) * ((x-1)*x+g*sq(1+x)))
        ) / (
            (g-1)*sq((1+g+x-g*x)*(x*x-1)*(1+g))
        )
        return -sqterm_der / (RI(2)*self.alpha*(self.gamma+1)*sqrterm.sqrt()*sqrterm)

    def f_prime_abs_log(self, x):
	RI = self.field
        return log(abs(self.f_prime(x)))

    def f_second(self, x):
    	RI = self.field
        w,a,g,S = self.omega, self.alpha, self.gamma, self.Sigma
        sqrterm = max(RI(0), w / sq(a*(g+1)*x) + ((1-w)*(1+2*(x-1/(g+1))/(1-x))*S/(1-sq(((g-1)*x-1)/g))))
        #print([x,sqrterm])
        sqterm_der = -RI(2)*w/ (sq(a*(g+1)*x)*x) + RI(2)*(
            sq(g)*(1+g)*S*(w-1)*(x-1 + (g*(x-1)-x) * ((x-1)*x+g*sq(1+x)))
        ) / (
            (g-1)*sq((1+g+x-g*x)*(x*x-1)*(1+g))
        )
        sqterm_der2 = ((w-RI(1))*(g*g*g* S))/((sq(g)-RI(1)) * cb(x-RI(1))) + \
                      (RI(8)*w)/(sq(a*(1 + g)) * (x**5)) + \
                      (g*S*(w-RI(1)))/((sq(g)-1)*cb(RI(1) + x)) - \
                      (sq(g-1) * g * (sq(g)+1) * S *(w-RI(1)))/((g+RI(1))*cb(g*x -1 - g - x))
        return (RI(3)/RI(2)*sqterm_der/sqrterm - sqterm_der2) / \
            (RI(2)*self.alpha*(self.gamma+1)*sqrterm.sqrt()*sqrterm)
