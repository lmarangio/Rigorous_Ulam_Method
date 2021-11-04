"""
Plotting eigenvalues of a map
"""

from __future__ import division
from assembler import *
from dynamic import *
from eigenerr import *
from sparse import *
from estimator import *
from time import time
from partition import step_function
from plotting import plot_spectrum

from sage.all import *

prec = 53
epsilon = 1e-10;

coarsePartition = 2**10
finePartition = 2**13

print "Computing dynamic with %d digits and epsilon=%g" % (prec, epsilon)

t = time()

#D = PerturbedFourxDynamic(prec=prec, j = 8)
D = MannevillePomeauDynamic(prec=prec)

print "Assembling coarse Ulam matrix."
P = assemble(D, coarsePartition, basis='hat', epsilon=epsilon, prec=prec, n_jobs=1)
print "Computing eigenvalues"
plot_spectrum(P).show()

