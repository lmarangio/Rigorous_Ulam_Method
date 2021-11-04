"""
Infinity-norm estimation. WIP.
"""

from __future__ import division
from assembler import *
from dynamic import *
from eigenerr import *
from sparse import *
from estimator import *
from time import time
from partition import equispaced
from plotting import *

from sage.all import *
import numpy as np

prec = 53
epsilon = 1e-10;

coarsePartition = 2**10

print "Computing dynamic with %d digits and epsilon=%g" % (prec, epsilon)

t = time()

D = PerturbedFourxDynamic(prec=prec, j = 8, c=0.01)
DD = D.iterate_with_lasota_yorke('Linf') #this should be equal to D if the example above is j=8,c=0.01

print "Assembling Ulam matrix."
P = assemble(DD, coarsePartition)

alphaC = 1/256
NC = decay_time(P, alpha=alphaC, norm='Linf')
print "Decay time (to alpha=%g): %d" % (alphaC,NC)

v = perron_vector(P)
v = v.real
v = v/v.sum()

res = rigorous_residual(P, v, norm='Linf')
print "Linf residual of the computed Perron vector: ", res.magnitude()


err = global_error(DD, coarsePartition, NC, alphaC, res, type='Linf')
print "Total Linf error: %g" % err

print "Total time taken: %g s." % (time() - t)

filename = "invariant.txt"
np.savetxt(filename, v)
print "The invariant measure has been saved in the file %s" % filename


plot = plot_dynamic(D) + plot_measure(v, coarsePartition, err, norm='Linf')
plot.show()

