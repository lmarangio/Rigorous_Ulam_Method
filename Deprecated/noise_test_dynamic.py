
import numpy as np
import os
import joblib
import plotting
from safe_interval_newton import *
from test1 import *

def test_derivative(RI, f, fprime, N):
    for i in range(N):
        a = RI(i)/N
        b = RI(i+1)/N
        d = fprime(a.union(b))
        inc = (f(b) - f(a)) / (b-a)

        # in theory this could fail and still the functions be correct...
        # in practice it is a good test for the derivative
        if not (inc in d):
            print a, b
            print inc
            print d
            return False
    return True

prec = 512
D = Test1(prec)
RI = D.field

# try to compute some preimage
D.preimage(D.field(0.5), 1, D.field(2).upper()**(100-D.field.prec()))

if not test_derivative(RI, D.f_plain, D.f_prime, 10000):
    raise ValueError, 'f_prime is not the derivative of f!'
if not test_derivative(RI, D.f_prime, D.f_second, 10000):
    raise ValueError, 'f_second is not the derivative of f_prime!'
print 'congrats, derivatives are OK!'

graphplot = plot(lambda x: D.f(RI(x)).center(), 0, 1, color="#000")
#graphplot = plot(lambda x: D.f_prime(RI(x)).center(), 0, 1, color="#000")

#print safe_interval_newton(lambda x:D.f_prime(x), lambda x:D.f_second(x), RI(0.7, 0.8), RI(0), 0.00001)

graphplot.show(ymin = 0, ymax = 1)
