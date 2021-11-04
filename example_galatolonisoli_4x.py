from __future__ import division
from dynamic import *
from hat_evaluation import *
from generic_assembler import *
from generic_estimator import *
from power_norms import *
from partition import equispaced
from eigenerr import perron_vector
from plotting import *
from sparse import *
import numpy as np
from injection import *

# Current version, on Luminous:
# error from new two-grid argument + extension (to m=40) = 0.00379640421532717
# Elapsed time: 1094.34163022 seconds.

import time
start = time.time()

D = PerturbedFourxDynamic(c=0.01, j=8)
DD = D.iterate_with_lasota_yorke('Linf') #TODO: once we move fully to DFLY inequalities, this should not be necessary
p = equispaced(8192)

integral_value = 1/512

basis = HatFunctionsInjectedWithEvaluationsLinf(p, integral_value)
m = 15
m_extend = 100

print "Assembling..."
P = basis.assemble(DD, 1e-12)
time1 = time.time() - start
start = time.time()
print "Done."
print "Computing power norms..."
M = basis.bound_on_norms_of_powers(DD, project_left=True, project_right=True)

Ci = norms_of_powers(sage_sparse_to_scipy_sparse(P), M, integral_value, m, K=interval_norminf_error(P))
time2 = time.time() - start
start = time.time()
print "Done."

# p_f = equispaced(524288)
#p_f = equispaced(2**16)
p_f = equispaced(2**16)

fine_basis = HatFunctionsInjectedWithEvaluationsLinf(p_f, integral_value)
print "Assembling fine grid..."
P_f = fine_basis.assemble(DD, 1e-12)
time3 = time.time() - start
start = time.time()
print "Done."
v_f = perron_vector(P_f)
v_f = fine_basis.sanitize_perron_vector(v_f)
residual_f = fine_basis.residual_estimate(P_f, v_f)

error_to_plot = None
#try:
Di = extend_power_norms(Ci, m_extend - 1)
Ci_f = power_norms_from_smaller_grid(DD, basis, fine_basis, Di)
Di_f = extend_power_norms(Ci_f)
error_f = error_bound_from_power_norms(DD, fine_basis, Di_f, residual_f)
error_to_plot = error_f
print "error from new two-grid argument + extension (to m=%s) = %s" % (m_extend, error_f)
#except ValueError:
#	print "error from new two-grid argument + extension (to m=%s) = FAILED" % (m_extend,)

time4 = time.time() - start
print "Assembly time: %s seconds." % time1
print "Power norms time: %s seconds." % time2
print "Fine assembly time: %s seconds." % time3
print "Perron + estimation time: %s seconds." % time4
print "Total time: %s seconds." % (time1+time2+time3+time4)

pl = plot_dynamic(D) + plot_measure(v_f, p_f, error_to_plot, basis='hat', norm='Linf')
pl.show()
