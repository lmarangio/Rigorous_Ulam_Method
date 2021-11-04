"""
Recreates the data for the plots in the two-grid L1 example on Federico's Householder slides
"""

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

D = PerturbedFourxDynamic(c=0.4, j=2)
DD = D.iterate_with_lasota_yorke('L1')

m = 15
m_extend = 45

def onegrid_example(n):

	start = time.time()
	p = equispaced(n)
	basis = UlamL1(p)
	P = basis.assemble(DD, 1e-12)
	stop = time.time()
	assemble_time = stop - start

	start = time.time()
	M = basis.bound_on_norms_of_powers(DD, project_left=True, project_right=True)
	Ci = norm1_of_powers(sage_sparse_to_scipy_sparse(P), M, m, K=interval_norm1_error(P))
	stop = time.time()
	Ci_time = stop - start
	print Ci

	start = time.time()
	v = perron_vector(P)

	v = basis.sanitize_perron_vector(v)
	residual = basis.residual_estimate(P, v)
	stop = time.time()
	residual_time = stop - start

	start = time.time()
	Di = extend_power_norms(Ci, m_extend)
	error = error_bound_from_power_norms(DD, basis, Di, residual)
	stop = time.time()
	estimation_time = stop - start
	total_time = assemble_time + Ci_time + residual_time + estimation_time
	print 'n: %s, error: %s, time: %s' % (n, error, total_time)
	return [n, assemble_time, Ci_time, residual_time, estimation_time, error, total_time]

def twogrid_example(n, n_f):
	start = time.time()
	p = equispaced(n)
	basis = UlamL1(p)
	P = basis.assemble(DD, 1e-12)
	stop = time.time()
	coarse_assemble_time = stop - start

	start = time.time()
	M = basis.bound_on_norms_of_powers(DD, project_left=True, project_right=True)
	Ci = norm1_of_powers(sage_sparse_to_scipy_sparse(P), M, m, K=interval_norm1_error(P))
	stop = time.time()
	Ci_time = stop - start
	print Ci

	start = time.time()
	p_f = equispaced(n_f)
	fine_basis = UlamL1(p_f)
	P_f = fine_basis.assemble(DD, 1e-12)
	stop = time.time()
	fine_assemble_time = stop - start
	assemble_time = coarse_assemble_time + fine_assemble_time

	start = time.time()
	v = perron_vector(P_f)
	v = fine_basis.sanitize_perron_vector(v)
	residual = fine_basis.residual_estimate(P_f, v)
	stop = time.time()
	residual_time = stop - start

	start = time.time()
	Di = extend_power_norms(Ci, m_extend)
	Ci_f = power_norms_from_smaller_grid(DD, basis, fine_basis, Di)
	Di_f = extend_power_norms(Ci_f)
	error = error_bound_from_power_norms(DD, fine_basis, Di_f, residual)
	stop = time.time()
	estimation_time = stop - start
	print Ci_f, Di_f
	total_time = assemble_time + Ci_time + residual_time + estimation_time
	print 'n: %s, error: %s, time: %s' % (n, error, total_time)
	return [n_f, assemble_time, Ci_time, residual_time, estimation_time, error, total_time, coarse_assemble_time, fine_assemble_time]

results = [onegrid_example(2**k) for k in range(10, 15)]  # TODO: want 15 as the end range in the end
with open('onegrid.dat', 'wb') as f:
	f.write('\t'.join(['n', 'assemble_time', 'Ci_time', 'residual_time', 'estimation_time', 'error', 'total_time']))
	f.write('\n')
	for r in results:
		f.write('\t'.join(map(str, r)))
		f.write('\n')

results = [twogrid_example(1024, 2**k) for k in range(10, 18)]  # TODO: want 18 as the end range in the end
with open('twogrid.dat', 'wb') as f:
	f.write('\t'.join(['n', 'assemble_time', 'Ci_time', 'residual_time', 'estimation_time', 'error', 
		'total_time', 'coarse_assemble_time', 'fine_assemble_time']))
	f.write('\n')
	for r in results:
		f.write('\t'.join(map(str, r)))
		f.write('\n')
