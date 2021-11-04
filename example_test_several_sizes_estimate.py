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
import pickle
import time

# WARNING: this dynamic has to match the one in example_test_several_sizes, for the results to be
# meaningful
D = PerturbedFourxDynamic(c=0.01, j=8)
DD = D.iterate_with_lasota_yorke('Linf') #TODO: once we move to DFLY inequalities, this should not be necessary


with open('test_several_sizes.pickle', 'rb') as f:
	results = pickle.load(f)

results = [x for x in results if x['n'] >= 1024]

ns = [x['n'] for x in results]
assemble_times = [x['assemble_time'] for x in results]
Ci_times = [x['Ci_time'] for x in results]
residual_times = [x['residual_time'] for x in results]

def twogrid_result(n, nf):
	"""
	Returns a triple (time, error, label) that can be obtained using the two-grid arguments
	with grids of size n, nf respectively, or the one-grid argument if nf == n
	"""

	if nf < n:
		raise ValueError, 'The fine grid has to be finer than the coarse grid'

	m_extend = 40

	if nf == n: #one-grid argument
		xx = [x for x in results if x['n']==n ][0]

		Ci = xx['Ci']
		Di = extend_power_norms(Ci, m_extend)
		basis = HatFunctionsWithEvaluationsLinf(equispaced(n))
		error_f = error_bound_from_power_norms(DD, basis, Di, xx['residual'])

		time = xx['assemble_time'] + xx['Ci_time'] + xx['residual_time']
		label = '%s' % n

	else:

		xx = [x for x in results if x['n']==n ][0]
		xf = [x for x in results if x['n']==nf ][0]

		Ci = xx['Ci']
		Di = extend_power_norms(Ci, m_extend)
		basis = HatFunctionsWithEvaluationsLinf(equispaced(n))
		fine_basis = HatFunctionsWithEvaluationsLinf(equispaced(nf))
		Ci_f = power_norms_from_smaller_grid(DD, basis, fine_basis, Di)
		Di_f = extend_power_norms(Ci_f)
		try:
			error_f = error_bound_from_power_norms(DD, fine_basis, Di_f, xf['residual'])
		except ValueError:
			return None

		# TODO: the time could be improved since we do not need to assemble the coarse matrix,
		# we can "aggregate" it from the fine one.

		time = xx['assemble_time'] + xx['Ci_time'] + xf['assemble_time'] + xf['residual_time']
		label = '(%s, %s)' % (n, nf)

	return (time, error_f, label)

twogrid_results = [twogrid_result(n, nf) for n in ns for nf in ns if nf >= n]

with open('test_several_sizes_twogrid_results.pickle', 'wb') as f:
	pickle.dump(twogrid_results, f)
