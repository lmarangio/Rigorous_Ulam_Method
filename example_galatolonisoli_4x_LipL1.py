from __future__ import division
from dynamic import *
from hat_evaluation_LipL1 import *
from generic_assembler import *
from generic_estimator import *
from power_norms import *
from partition import equispaced
from eigenerr import perron_vector
from plotting import *
from sparse import *
import numpy as np
from injection import *
from warnings import warn

import time
start = time.time()

D = PerturbedFourxDynamic(c=0.01, j=8)
DD = D.iterate_with_lasota_yorke('Linf') #TODO: once we move fully to DFLY inequalities, this should not be necessary
p = equispaced(1024)

coarse_basis = HatFunctionsWithEvaluationsLipL1(p)
m = 15
m_extend = 100

print("Assembling...")
P = coarse_basis.assemble(DD, 1e-12)
time1 = time.time() - start
start = time.time()
print("Done.")
print("Computing power norms...")

PP = sage_sparse_to_scipy_sparse(P)
K = interval_norminf_error(P)
Ki = norminf_of_powers_normalized_naive(P, m)

(lips, norms) = coarse_basis.small_matrix_bounds(DD, m)

bounds = vector(RealField(rnd='RNDU'), m)

warn('TODO: check if norms[i] can be used here or it''s in the wrong norm')
for i in range(m): #combines known bounds on ||Q_h^i||
	bounds[i] = min(Ki[i], norms[i])

Ci = norminf_of_powers_normalized(PP, bounds, m, K)
Ci = vector(RealField(rnd='RNDU'), Ci)

time2 = time.time() - start
start = time.time()
print("Done.")

Di = extend_power_norms(Ci, m_extend - 1)

v = perron_vector(P)
v = coarse_basis.sanitize_perron_vector(v)
residual = coarse_basis.residual_estimate(P, v)
error = error_bound_from_power_norms(DD, coarse_basis, Di, residual)

print("error from one-grid argument + extension (to m=%s) = %s" % (m_extend, error))

# p_f = equispaced(524288)
p_f = equispaced(2**16)
#p_f = equispaced(2**17)

fine_basis = HatFunctionsWithEvaluationsLipL1(p_f)
print("Assembling fine grid...")
P_f = fine_basis.assemble(DD, 1e-12)
PP_f = sage_sparse_to_scipy_sparse(P_f)
time3 = time.time() - start
start = time.time()
print("Done.")
v_f = perron_vector(P_f)
v_f = fine_basis.sanitize_perron_vector(v_f)
residual_f = fine_basis.residual_estimate(P_f, v_f)

error_to_plot = None
#try:
Di = extend_power_norms(Ci, m_extend)

# complicated list of conversions to compute this norm
norm_fineP = matrix_norminf(sage_sparse_to_scipy_sparse(P_f, lambda x: x.magnitude()))

Ci_f = power_norms_from_smaller_grid_alternate(DD, coarse_basis, fine_basis, Di, norm_fineP)
Ci_f = vector(RealField(rnd='RNDU'), Ci_f)

(lips_f, norms_f) = fine_basis.small_matrix_bounds(DD, m_extend)
K_f = interval_norminf_error(P_f)
Ki_f = norminf_of_powers_normalized_naive(P_f, m_extend)

bounds_f = vector(RealField(rnd='RNDU'), m_extend)
warn('TODO: check if norms[i] can be used here or it''s in the wrong norm')
for i in range(len(Ci_f)):
	bounds_f[i] = min([Ci_f[i], norms_f[i], Ki_f[i]])

Di_f = extend_power_norms(bounds_f)
error_f = error_bound_from_power_norms(DD, fine_basis, Di_f, residual_f)
error_to_plot = error_f
print("error from new two-grid argument + extension (to m=%s) = %s" % (m_extend, error_f))
#except ValueError:
#	print "error from new two-grid argument + extension (to m=%s) = FAILED" % (m_extend,)

time4 = time.time() - start
print("Assembly time: %s seconds." % time1)
print("Power norms time: %s seconds." % time2)
print("Fine assembly time: %s seconds." % time3)
print("Perron + estimation time: %s seconds." % time4)
print("Total time: %s seconds." % (time1+time2+time3+time4))

pl = plot_dynamic(D) + plot_measure(v_f, p_f, error_to_plot, basis='hat', norm='Linf')
pl.show()
