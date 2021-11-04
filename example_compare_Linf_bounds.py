from __future__ import division
from dynamic import *
from hat_evaluation import *
from generic_assembler import *
from generic_estimator import *
from power_norms import *
from partition import equispaced
from eigenerr import perron_vector
from plotting import *
import numpy as np
from eigenerr import *

D = PerturbedFourxDynamic(c=0.25)
DD = D.iterate_with_lasota_yorke('Linf') #TODO: once we move to DFLY inequalities, this should not be necessary
p = equispaced(1024)
basis = HatFunctionsWithEvaluationsLinf(p)
m = 10

print "Assembling..."
P = assemble(DD, basis, 1e-12)
print "Done."
M = basis.bound_on_norms_of_powers(DD, project_left=True, project_right=True)

print "Computing power norms..."
Ci = norminf_of_powers(sage_sparse_to_scipy_sparse(P), M, m, K=interval_norm1_error(P))
print "Done."
print Ci

print "Computing power norms with the (non-rigorous) alternate method..."
Ci = norminf_of_powers_alternate(sage_sparse_to_scipy_sparse(P), M, m, K=interval_norm1_error(P))
print "Done."
print Ci


for alpha in [0.5, 0.25, 0.1, 0.005, 0.004, 0.003]:
	print "Computing decay time to %s..." % alpha
	NC = decay_time(P, alpha=alpha, norm='Linf')
	print NC
	print "Done."
