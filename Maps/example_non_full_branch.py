"""
Example, use the two-grid approach to obtain a guaranteed bound on a finer grid through the decay time on a coarser grid.
"""

from __future__ import division
from assembler import *
from dynamic import *
from eigenerr_hole import *
from sparse import *
from estimator import *
from time import time
from partition import step_function
from scipy.sparse.linalg import eigs

from sage.all import *
import numpy as np

prec = 53
epsilon = 1e-10

#coarsePartition = 2**10

finePartition = 2**13

print "Computing dynamic with %d digits and epsilon=%g" % (prec, epsilon)

t = time()

D = QuotientedLinearDynamic(Rational('23/5'), prec=prec)

#print "Assembling coarse Ulam matrix."
#P = assemble(D, coarsePartition, epsilon=epsilon, prec=prec, n_jobs=3)


#(NF, alphaF) = optimal_twogrid_decay_time(D, coarsePartition, finePartition, NC, alphaC)
#print "Estimated decay time (to alpha=%g) on a grid of size 1/%d: %d" % (alphaF, finePartition, NF)

#print "Assembling fine Ulam matrix."
P = assemble(D, finePartition, epsilon=epsilon, prec=prec, n_jobs=1)

w = P.parent().column_space()(0)

v = perron_vector(P)

PP = sage_sparse_to_scipy_sparse(P.apply_map(lambda x: x.center()))
L, V = eigs(PP, 30)
#L, V = np.linalg.eig(PP.todense())

alphaC = 1/2
#NC = decay_time_hole(P, alpha=alphaC)
#print "Decay time (to alpha=%g): %d" % (alphaC,NC)



print "Total time taken: %g s." % (time() - t)
scatter_plot([(real(x), imag(x)) for x in L]).show()


res = rigorous_residual(P, v)
print "L1 residual of the computed Perron vector: ", res.magnitude()

#res2 = global_error(D, finePartition, NF, alphaF, res)
#print "Total norm-1 error: %g" % res2 

#print "Error obtained if we had used the coarse grid only (estimated): %g" % global_error(D, coarsePartition, NC, alphaC, res)

print "Total time taken: %g s." % (time() - t)

filename = "invariant_hole.txt"
np.savetxt(filename,v)
#print "The invariant measure has been saved in the file %s" % filename

pl = plot(lambda x: D.f(RIF(x)).center(), 0, 1, legend_label='Map T') + \
	plot_step_function(step_function(v, finePartition), color='red', legend_label='Invariant measure')# + \
	#bar_chart([sqrt(res2)], width=sqrt(res2), color='green', legend_label='Area of the total error')
pl.show()

