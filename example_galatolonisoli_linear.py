from __future__ import division
from dynamic import *
from ulam import *
from generic_assembler import *
from generic_estimator import *
from power_norms import *
from partition import equispaced
from eigenerr import perron_vector
from plotting import *
import numpy as np

# on Axiom:
# error from new two-grid argument + extension (to m=100) = 0.00178910650819843
# Elapsed time: 73.9266078472 seconds.
# On luminous: 147.409 seconds

import time
start = time.time()

D = QuotientedLinearDynamic(RIF(Rational('17/5')))
p = equispaced(1024)
basis = UlamL1(p)
m = 40
m_extend = 100

print "Assembling..."
P = assemble(D, basis, 1e-12)
print "Done."
print "Computing power norms..."
M = basis.bound_on_norms_of_powers(D, project_left=True, project_right=True)
Ci = norm1_of_powers(sage_sparse_to_scipy_sparse(P), M, m, K=interval_norm1_error(P))
print "Done."

# two-grid stuff

p_f = equispaced(131072)
fine_basis = UlamL1(p_f)
print "Assembling fine grid..."
P_f = assemble(D, fine_basis, 1e-12)
print "Done."
v_f = perron_vector(P_f)
v_f = fine_basis.sanitize_perron_vector(v_f)
residual_f = fine_basis.residual_estimate(P_f, v_f)

Di = extend_power_norms(Ci, m_extend)
Di_f = extend_power_norms(power_norms_from_smaller_grid(D, basis, fine_basis, Di))
error_f = error_bound_from_power_norms(D, fine_basis, Di_f, residual_f)
print "error from new two-grid argument + extension (to m=%s) = %s" % (m_extend, error_f)

end = time.time()
print "Elapsed time: %s seconds." % (end - start)

plot = plot_dynamic(D) + plot_measure(v_f, p_f, error_f)
plot.show()
