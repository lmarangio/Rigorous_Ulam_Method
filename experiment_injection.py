from __future__ import division
from dynamic import *
from injection import *
from generic_assembler import *
from generic_estimator import *
from power_norms import *
from partition import equispaced
from eigenerr import perron_vector
from plotting import *
from sparse import *
import numpy as np

import time
start = time.time()

D = PerturbedFourxDynamic(c=0.001, j=4)
DD = D.iterate_with_lasota_yorke('Linf') #TODO: once we move fully to DFLY inequalities, this should not be necessary
n = 1024
n_f = 4096
p = equispaced(n)

integral_value = 2.

basis = HatFunctionsInjectedWithEvaluationsLinf(p, integral_value)
m = 10
m_extend = 50

print "Assembling..."
P = basis.assemble(DD, 1e-12)
print "Done."
print "Computing power norms..."
M = basis.bound_on_norms_of_powers(DD, project_left=True, project_right=True)

Ci = norms_of_powers(sage_sparse_to_scipy_sparse(P), M, integral_value, m, K=interval_norm_error(P, integral_value))
print "Done."
Di = extend_power_norms(Ci, m_extend)

v = basis.sanitize_perron_vector(perron_vector(P))

residual = basis.residual_estimate(P, v)
error = error_bound_from_power_norms(DD, basis, Di, residual)
print "error from one-grid argument + extension (to m=%s) = %s" % (m_extend, error)

p_f = equispaced(n_f)

fine_basis = HatFunctionsInjectedWithEvaluationsLinf(p_f, integral_value)
print "Assembling fine grid..."
P_f = fine_basis.assemble(DD, 1e-12)
print "Done."
v_f = perron_vector(P_f)
v_f = fine_basis.sanitize_perron_vector(v_f)
residual_f = fine_basis.residual_estimate(P_f, v_f)

#try:
Ci_f = power_norms_from_smaller_grid(DD, basis, fine_basis, Di)
Di_f = extend_power_norms(Ci_f)
error_f = error_bound_from_power_norms(DD, fine_basis, Di_f, residual_f)
error_to_plot = error_f
print "error from new two-grid argument + extension (to m=%s) = %s" % (m_extend, error_f)
#except ValueError:
#	print "error from new two-grid argument + extension (to m=%s) = FAILED" % (m_extend,)

end = time.time()
print "Elapsed time: %s seconds." % (end - start)

pl = plot_dynamic(D) + plot_measure(v_f, p_f, error_to_plot, basis='hat', norm='Linf')
pl.show()
