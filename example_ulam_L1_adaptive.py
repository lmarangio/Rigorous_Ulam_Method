"""
Example: change partition lengths iteratively
"""


from __future__ import division

from assembler import *
from dynamic import *
from eigenerr import *
from sparse import *
from estimator import *
from partition import equispaced
from plotting import *
from refiner import *

from sage.all import *
import numpy as np


prec = 53
epsilon = 1e-10


n1 = 2**10
n2 = 2**14
coarsePartition = equispaced(n1)

print "Computing dynamic with %d digits and epsilon=%g" % (prec, epsilon)

# The Mongeville map is the only one where this matters

#D = FourxDynamic(prec=prec, epsilon=epsilon/2)
#D = MannevillePomeauDynamic(prec=prec, epsilon=epsilon/2)
#D = PerturbedFourxDynamic(prec=prec, c = 0.4)
D = MongevillePoleauDynamic(prec=prec, beta=3/4, alpha=2)

print "Assembling matrix"
PC = assemble(D, coarsePartition, epsilon=epsilon, prec=prec, n_jobs=4)

vC = perron_vector(PC)
# Sanitize
vC = np.real(vC)
vC = vC / sum(vC)
vC[vC<0] = 0
vC = vC / np.linalg.norm(vC, ord=1)

res = rigorous_residual(PC, vC)
print "L1 residual of the computed Perron vector: ", res.magnitude()

alphaC = 1/256
NC = decay_time(PC, alpha=alphaC)
print "Decay time (to alpha=%g): %d" % (alphaC, NC)

errC = global_error(D, coarsePartition, NC, alphaC, res)
print "The total norm-1 error on the computed v is %g" % errC

print "Refining partition adaptively (to about %d intervals)." % n2
finePartition = refine(partition_sqrt(vC, coarsePartition), coarsePartition, n2)

print "Assembling matrix"
PF = assemble(D, finePartition, epsilon=epsilon, prec=prec, n_jobs=4)

vF = perron_vector(PF)
# Sanitize
vF = np.real(vF)
vF = vF / sum(vF)
vF[vF<0] = 0
vF = vF / np.linalg.norm(vF, ord=1)

resF = rigorous_residual(PF, vF)
print "L1 residual of the computed Perron vector: ", resF.magnitude()

(NF, alphaF) = optimal_twogrid_decay_time(D, coarsePartition, finePartition, NC, alphaC)
print "Estimated decay time (to alpha=%g) of the fine grid: %d" % (alphaF, NF)

errFg = global_error(D, finePartition, NF, alphaF, resF)
print "Total norm-1 error on the computed w (via the SUBOPTIMAL global error bound): %g" % errFg

errF = localized_error(D, vC, coarsePartition, errC, finePartition, NF, alphaF, resF)
print "Total norm-1 error on the computed w (via the better local error bound): %g" % errF

plot = plot_dynamic(D) + plot_measure(vF, finePartition, errF)
plot.show()

