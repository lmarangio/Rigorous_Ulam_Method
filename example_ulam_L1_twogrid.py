"""
Example, use the two-grid approach to obtain a guaranteed bound on a finer grid through the decay time on a coarser grid.
"""

from __future__ import division
from assembler import *
from dynamic import *
from eigenerr import *
from sparse import *
from estimator import *
from time import time
from plotting import *

from sage.all import *
import numpy as np

prec = 53
epsilon = 1e-10;

coarsePartition = 2**10
finePartition = 2**17

print "Computing dynamic with %d digits and epsilon=%g" % (prec, epsilon)

t = time()

D = PerturbedFourxDynamic(prec=prec, c = 0.4)
#D = MongevillePoleauDynamic(prec=prec,beta=1/4)

print "Assembling coarse Ulam matrix."
P = assemble(D, coarsePartition, epsilon=epsilon, prec=prec, n_jobs=3)

alphaC = 1/256
NC = decay_time(P, alpha=alphaC)
print "Decay time (to alpha=%g): %d" % (alphaC,NC)

(NF, alphaF) = optimal_twogrid_decay_time(D, coarsePartition, finePartition, NC, alphaC)
print "Estimated decay time (to alpha=%g) on a grid of size 1/%d: %d" % (alphaF, finePartition, NF)

print "Assembling fine Ulam matrix."
P = assemble(D, finePartition, epsilon=epsilon, prec=prec, n_jobs=1)

v = perron_vector(P)
# Sanitize
v = np.real(v)
v = v / sum(v)
v[v<0] = 0
v = v / np.linalg.norm(v, ord=1)

res = rigorous_residual(P, v)
print "L1 residual of the computed Perron vector: ", res.magnitude()

err = global_error(D, finePartition, NF, alphaF, res)
print "Total norm-1 error: %g" % err

print "Error obtained if we had used the coarse grid only (estimated): %g" % global_error(D, coarsePartition, NC, alphaC, res)

print "Total time taken: %g s." % (time() - t)

filename = "invariant.txt"
np.savetxt(filename,v)
print "The invariant measure has been saved in the file %s" % filename

plot = plot_dynamic(D) + plot_measure(v, finePartition, err)
plot.show()
