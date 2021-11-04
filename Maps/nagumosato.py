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


class NagumoSato(Dynamic):
    def __init__(self,
                 prec,
                 alpha = '0.5',
                 lambd = '0.5666666666666666666666666666666666666666666666666666666',
                 beta = '120'):
        self.name = "nagumosato"
        
        RI = self.field = RealIntervalField(prec=prec)
        R = self.real_field = RealField(prec=prec)

        self.alpha = RI(alpha)
        self.lambd = RI(lambd)
        self.beta  = RI(beta)

        self.flex = (RI('1') - self.lambd) / self.alpha

        self.sep1 = (-log((-(2-self.beta) + sqrt((self.beta-4)*self.beta)) / 2) / self.beta
                     + 1 - self.lambd) / self.alpha
        self.sep2 = (-log((-(2-self.beta) - sqrt((self.beta-4)*self.beta)) / 2) / self.beta
                     + 1 - self.lambd) / self.alpha
        #eps = RI(2).center()**(5-self.field.prec())
        #self.sep1 = safe_interval_newton(self.f_prime, self.f_second,
        #                                 RI(0, self.flex.lower()), RI(0), eps)
        #self.sep2 = safe_interval_newton(self.f_prime, self.f_second,
        #                                 RI(self.flex.upper(),1), RI(0), eps)

        self._domain = RI('0','1')
        self.nbranches = 3
        
        self.branch_intervals = [ (RI('0'), self.sep1),
                                  (self.sep1, self.sep2),
                                  (self.sep2, RI('1')) ]
        self.branch_enclosures = [ self.sep1.union(RI('0')),
                                   self.sep1.union(self.sep2),
                                   self.sep2.union(RI('1')) ]
        self.branch_increasing = [True, False, True]

    def f(self, x):
        RI = self.field
        X = exp(-self.beta * (self.alpha*x + self.lambd - 1))
        return self.alpha*x + self.lambd - 1 / (1 + X)

    def f_prime(self, x):
        RI = self.field
        F = (-self.beta * self.alpha)
        X = exp(-self.beta * (self.alpha*x + self.lambd - 1))
        return self.alpha + F * X / (1 + X)**2

    def f_second(self, x):
        RI = self.field
        F = (-self.beta * self.alpha)
        X = exp(-self.beta * (self.alpha*x + self.lambd - 1))
        return F ** 2 * X / (1 + X)**2 * (1  -  2 * X / (1 + X))
        
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
