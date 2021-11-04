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
m = 10

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

plot = plot_dynamic(D) + plot_measure(v, p, error)
plot.show()
