from __future__ import division
from dynamic import *
from ulam import *
from generic_assembler import *
from generic_estimator import *
from decay import *
from partition import equispaced
from eigenerr import perron_vector
from plotting import *

D = PerturbedFourxDynamic(c=0.3)
p = equispaced(1024)
basis = UlamL1(p)

alpha=0.5

print "Assembling..."
P = assemble(D, basis, 1e-12)
print "Done."
print "Computing decay time..."
n = decay_time(D, basis, P, alpha = alpha)
print "Done."

#import eigenerr
#n2 = eigenerr.decay_time(P, alpha,1,'L1')
#assert n2 == n

v = perron_vector(P)

v = basis.sanitize_perron_vector(v)
residual = basis.residual_estimate(P, v)

print "residual = %g" % residual

error = error_bound_from_decay_time(D, basis, n, alpha, residual)
print "error = %s" % error

# two-grid
p_f = equispaced(8192)
alpha_c = 1/256
n_c = decay_time(D, basis, P, alpha_c)
fine_basis = UlamL1(p_f)
P_f = assemble(D, fine_basis, 1e-12)
v_f = perron_vector(P_f)
v_f = basis.sanitize_perron_vector(v_f)
residual_f = basis.residual_estimate(P_f, v_f)
(n_f, alpha_f) = decay_time_estimate_from_smaller_grid(D, basis, fine_basis, n_c, alpha_c)
error_f = error_bound_from_decay_time(D, fine_basis, n_f, alpha_f, residual)

plot = plot_dynamic(D) + plot_measure(v_f, p_f, error_f)
plot.show()
