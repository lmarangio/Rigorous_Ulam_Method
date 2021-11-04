"""
Basic example, discretize on a grid and estimate.
"""

from __future__ import division

from assembler import *
from dynamic import *
from eigenerr import *
from sparse import *
from estimator import *
from sage.all import *
from plotting import *
import numpy as np


prec = 53
epsilon = 1e-10

# a non-homogeneous partition, just to check if everything works
k = 10;
partition = [i*(0.5 ** (k+1)) for i in range(2 ** k)] + [0.5 + i*(0.5 ** k) for i in range(2 ** (k-1)+1)]
partition = np.asarray(partition)

print "Computing dynamic with %d digits and epsilon=%g" % (prec, epsilon)

#D = FourxDynamic(prec=prec, epsilon=epsilon/2)
#D = MannevillePomeauDynamic(prec=prec, epsilon=epsilon/2)
#D = PerturbedFourxDynamic(prec=prec, c = 0.4)
#D = MongevillePoleauDynamic(prec=prec)
D = LanfordDynamic(prec=prec)

print "Assembling Ulam matrix."
P = assemble(D, partition, epsilon=epsilon, prec=prec, n_jobs=4)

v = perron_vector(P)

# Sanitize
v = np.real(v)
v = v / sum(v)
v[v<0] = 0
v = v / np.linalg.norm(v, ord=1)

res = rigorous_residual(P, v)
print "L1 residual of the computed Perron vector: ", res.magnitude()

alpha = 1/2
N = decay_time(P, alpha=alpha)
print "Decay time (to alpha=%g): %d" % (alpha,N)

err = global_error(D, partition, N, alpha, res)
print "The total norm-1 error on the computed v is %g" % err

filename = "invariant.txt"
np.savetxt(filename,v)
print "The invariant measure has been saved in the file %s" % filename

plot = plot_dynamic(D) + plot_measure(v, partition, err)
plot.show()

