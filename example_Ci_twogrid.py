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

D = PerturbedFourxDynamic(c=0.3)
p = equispaced(1024)
basis = UlamL1(p)
m = 20
m_extend = 40

print "Assembling..."
P = assemble(D, basis, 1e-12)
print "Done."
print "Computing power norms..."
M = basis.bound_on_norms_of_powers(D, project_left=True, project_right=True)
Ci = norm1_of_powers(sage_sparse_to_scipy_sparse(P), M, m, K=interval_norm1_error(P))
print "Done."

#import eigenerr
#n2 = eigenerr.decay_time(P, alpha,1,'L1')
#assert n2 == n

v = perron_vector(P)

v = basis.sanitize_perron_vector(v)
residual = basis.residual_estimate(P, v)

print "residual = %g" % residual

error = error_bound_from_power_norms(D, basis, Ci, residual)
print "error from Ci's = %s" % error

# simulates the corresponding bound obtained with the decay time
alpha = 0.5
n = np.argmax(Ci < alpha)
error2 = error_bound_from_decay_time(D, basis, n, alpha, residual)
print "error from decay time to %s = %s" % (alpha, error2)

# two-grid stuff

p_f = equispaced(8192)
fine_basis = UlamL1(p_f)
print "Assembling fine grid..."
P_f = assemble(D, fine_basis, 1e-12)
print "Done."
v_f = perron_vector(P_f)
v_f = fine_basis.sanitize_perron_vector(v_f)
residual_f = fine_basis.residual_estimate(P_f, v_f)

(n_f, alpha_f) = decay_time_estimate_from_smaller_grid(D, basis, fine_basis, n, alpha)
error_f2 = error_bound_from_decay_time(D, fine_basis, n_f, alpha_f, residual)
print "error from old two-grid argument = %s" % error_f2

Ci_f = power_norms_from_smaller_grid(D, basis, fine_basis, Ci)
error_f = error_bound_from_power_norms(D, fine_basis, Ci_f, residual)
print "error from new two-grid argument = %s" % error_f

Di = extend_power_norms(Ci, m_extend)
Di_f = extend_power_norms(power_norms_from_smaller_grid(D, basis, fine_basis, Di))
error_f = error_bound_from_power_norms(D, fine_basis, Di_f, residual)
print "error from new two-grid argument + extension (to m=%s) = %s" % (m_extend, error_f)


plot = plot_dynamic(D) + plot_measure(v_f, p_f, error_f)
plot.show()
