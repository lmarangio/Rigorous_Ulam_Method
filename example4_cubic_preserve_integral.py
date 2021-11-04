"""
Infinity-norm estimation. WIP.
"""

from __future__ import division
from assembler import *
from dynamic import *
from uniform_LY import *
from eigenerr import *
from sparse import *
from estimator import *
from time import time
from partition import step_function
from scipy.sparse.linalg import eigs
from basis_primitives import *

from sage.all import *
import numpy as np

prec = 53
epsilon = 1e-10;

partition = 2**10


print "Computing dynamic with %d digits and epsilon=%g" % (prec, epsilon)

t = time()

D = PerturbedFourxDynamic(prec=prec, c='0.03', j = 8)

DD = IterateMod1Dynamic(D,4)

#D=FourxDynamic(prec=prec)

pl=plot(lambda x: D.f(RIF(x)).center(), 0, 1, legend_label='Map T')+\
	plot(lambda x: D.fprime(RIF(x)).center(), 0, 1, legend_label='Map Tder',color='red')+\
	plot(lambda x: D.fsecond(RIF(x)).center(), 0, 1, legend_label='Map Tderder',color='green')
pl.show()

print "Assembling coarse Ulam matrix."
P = assemble(DD, partition, basis='Linear_cubic_preserve_integral', epsilon=epsilon, prec=prec, n_jobs=1)
s = last_row(P)
k = P.nrows()
P[ k-1 , : ] = s

PP = sage_sparse_to_scipy_sparse(P.apply_map(lambda x: x.center()))

time=eigenerr_Liverani(P,0.125)

w=np.zeros(coarsePartition+2)
w[coarsePartition+1]=1

for i in range(10):
	w=prod_mat_vec(PP,w)

z=from_derivative_at_nodes_to_compact_liverani_basis(w)

pl=plot(lambda x: evaluate_vector_compact_basis_Liverani(RIF(x),z).center(), 0, 1)

pl.show()



#time_to_alpha=decay_time(P,norm='Linf_C1')


#print "Testing it preserves the integral."

#error=test_matrix(PP)

#print error

"""

print "Computing eigenvalues"

L, V = eigs(PP, 30)
#L, V = np.linalg.eig(PP.todense())

print "Total time taken: %g s." % (time() - t)
scatter_plot([(real(x), imag(x)) for x in L]).show()

v=perron_vector(P)
test_plot_vector_corr(v)
"""
#alphaC = 1/256
#NC = decay_time_L1_canonical(P, alpha=alphaC)
#print "Decay time (to alpha=%g): %d" % (alphaC,NC)

#(NF, alphaF) = optimal_twogrid_decay_time(D, coarsePartition, finePartition, NC, alphaC)
#print "Estimated decay time (to alpha=%g) on a grid of size 1/%d: %d" % (alphaF, finePartition, NF)

#print "Assembling fine Ulam matrix."
#P = assemble(D, finePartition, epsilon=epsilon, prec=prec, n_jobs=1)

#v = perron_vector(P)
#res = rigorous_residual(P, v)
#print "L1 residual of the computed Perron vector: ", res.magnitude()
#
#res2 = global_error(D, finePartition, NF, alphaF, res)
#print "Total norm-1 error: %g" % res2 

#print "Error obtained if we had used the coarse grid only (estimated): %g" % global_error(D, coarsePartition, NC, alphaC, res)

#print "Total time taken: %g s." % (time() - t)

#filename = "invariant.txt"
#np.savetxt(filename,v)
#print "The invariant measure has been saved in the file %s" % filename

#pl = plot(lambda x: D.f(RIF(x)).center(), 0, 1, legend_label='Map T') + \
#	plot_step_function(step_function(v, finePartition), color='red', legend_label='Invariant measure') + \
#	bar_chart([sqrt(res2)], width=sqrt(res2), color='green', legend_label='Area of the total error')
#pl.show()

