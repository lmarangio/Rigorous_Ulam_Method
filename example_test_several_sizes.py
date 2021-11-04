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
import time
import pickle

D = PerturbedFourxDynamic(c=0.01, j=8)
DD = D.iterate_with_lasota_yorke('Linf') #TODO: once we move to DFLY inequalities, this should not be necessary

m = 10

def test_one_size(n, m):
	"""
	Gather some data on a specific grid size
	"""
	p = equispaced(n)
	basis = HatFunctionsWithEvaluationsLinf(p)

	start = time.time()
	P = assemble(DD, basis, 1e-12)
	stop = time.time()
	assemble_time = stop - start

	M = basis.bound_on_norms_of_powers(DD, project_left=True, project_right=True)

	start = time.time()
	Ci = norminf_of_powers_alternate(sage_sparse_to_scipy_sparse(P), M, m, K=interval_norm1_error(P))
	stop = time.time()
	Ci_time = stop - start

	start = time.time()
	v = perron_vector(P)

	v = basis.sanitize_perron_vector(v)
	residual = basis.residual_estimate(P, v)
	stop = time.time()
	residual_time = stop - start

	error = error_bound_from_power_norms(DD, basis, Ci, residual)
	return {'n': n, 'Ci': Ci, 'residual': residual, 'error': error, 
		'assemble_time': assemble_time, 'Ci_time': Ci_time, 'residual_time': residual_time}

results = [test_one_size(2**k, m) for k in range(4, 18)]

with open('test_several_sizes.pickle', 'wb') as f:
	pickle.dump(results, f)

