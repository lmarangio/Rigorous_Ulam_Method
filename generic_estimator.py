"""
Norm-agnostic functions for getting error estimates
"""

from itertools import count
from sage.all import vector, RealField
import numpy as np
from partition import is_refinement

def power_norms_from_uniform_bound(M, m):
	"""
	Obtain a Ci sequence from a uniform bound M

	Ci = [1., M, M, M, ..., M]
	"""

	Ci = vector(RealField(rnd='RNDU'), m)

	Ci[0] = 1.
	Ci[1:n] = M
	return Ci

def error_bound_from_decay_time(D, basis, m, alpha, residual):
	"""
	Estimates the error on the computed invariant measure using the computed decay time
	"""

	if not alpha < 1:
		raise ValueError('alpha has to be smaller than 1 (true contraction)')

	M = basis.bound_on_norms_of_powers(D, project_left=True, project_right=True)
	Ci = power_norms_from_uniform_bound(M, m+1)
	Ci[-1] = alpha
	return error_bound_from_power_norms(D, basis, Ci, residual)
	
def error_bound_from_power_norms(D, basis, Ci, residual):
	"""
	Estimates the error on the computed invariant measure using the bounds on \|L_\delta^i\|

	Input:
		D: dynamic
		basis: basis to use for the bounds
		Ci (sequence): bounds on powers of the discretized operator, 
			starting from Ci[0]=\|L_\delta^0\|=1. It must hold that Ci[-1] < 1. 
		residual: residual \|L_\delta v - v\| for the computed v. (with \|v\|=1).
	"""

	if not Ci[-1] < 1:
		raise ValueError('the last element of Ci must be <1')

	Ci = vector(D.field, Ci) #makes sure the Ci's are intervals
	amplification_factor = sum(Ci[:-1]) / (D.field(1)-Ci[-1])
	discretization_error = basis.invariant_measure_strong_norm_bound(D) * basis.mixed_matrix_norm_approximation_error(D)
	error = (discretization_error + residual) * amplification_factor
	return error.upper()

def error_bound_from_power_norms2(D, basis, Ci, residual):
	"""
	A version of error_bound_from_power_norms that uses the
	tighter bound in nonzerointegral.pdf (considering two
	separate norms for the discrete and continuous space and
	explicit conversion factors) 
	"""
	raise NotImplementedError

def minimize_on_integers(f, xmin, xmax):
	"""
	Find argmin(f) in range(xmin,xmax).
	
	Brutal implementation by trial and error.
	"""
	minvalue = float('inf')
	for x in xrange(xmin,xmax):
		fx = f(x)
		if fx < minvalue:
			argmin = x
			minvalue = fx
	
	return argmin

def first_below(f, M):
	"""
	Returns the first nonnegative integer `n` such that :math:`f(n) \leq M`.

	Args:
		f (int):
		M (real or interval):
	"""

	for n in count():
		if f(n) < M:
			return n

def decay_time_estimate_from_smaller_grid(D, coarse_basis, fine_basis, coarse_n, coarse_alpha, fine_alpha=None):
	"""
	Decay time estimate from the decay time on a smaller grid.
	
	If `fine_alpha` is provided, we use it as a target value. Otherwise, we choose 
	the pair (fine_n, fine_alpha) that minimizes (approximately) the coefficient `fine_n/(1-fine_alpha)`
	(which is equivalent to minimizing the error when we later call
	:func:`error_bound_from_decay_time(D, basis, fine_n, fine_alpha, residual)`).

	Returns:
		(fine_n, fine_alpha):
	"""

	delta = coarse_basis.mixed_matrix_norm_distance(fine_basis)
	
	MC = coarse_basis.bound_on_norms_of_powers(D, project_left=True, project_right=False)
	MF = fine_basis.bound_on_norms_of_powers(D, project_left=False, project_right=True)

	l, B = fine_basis.semidiscrete_lasota_yorke_constants(D)

	ll = 1 / (1-l)
	BB = B * ll * MC

	# constants such that :math:`\|L_F^(N+K)g\|_w \leq (C1 + C2\lambda^K)\|g\|_w`
	# see our notes: localization.tex, section "Coarse to fine II - the revenge"
	C1 = MF*(coarse_alpha + BB * delta*(coarse_n+1+ll))
	C2 = MC*delta*ll * fine_basis.strong_to_weak_norm_equivalence()

	f = lambda k: C1 + C2 * (l**k)

	if fine_alpha is None:
		if not C1 < 1:
			raise ValueError("Insufficient decay time estimate. Retry with a finer partition or a smaller coarse_alpha.")

		mink = first_below(f, 1)
		maxk = max(100,5*mink) #crude ballpark heuristic TODO: can we do better?
		objective_function = lambda k: (coarse_n + k) / (1 - f(k).center())
		k = minimize_on_integers(objective_function, mink, maxk)
		return coarse_n + k, f(k).magnitude()
	else:
		if not C1 < fine_alpha:
			raise ValueError("Insufficient decay time estimate. Retry with a finer partition or a smaller coarse_alpha.")
		k = first_below(f, fine_alpha)
		return coarse_n + k, fine_alpha

def power_norms_from_smaller_grid(D, coarse_basis, fine_basis, coarse_Ci):
	"""
	Estimates power norms Ci for a finer basis using those for a coarser basis

	Uses method in Isaia's working notes "estimates with Ci"

	Returns:
		fine_Ci: numpy vector of estimates
	"""

	if not is_refinement(fine_basis.partition, coarse_basis.partition):
		raise NotImplementedError('Implemented only for grids which are refinements')

	coarse_Ci = vector(D.field, coarse_Ci) #makes sure the Ci's are intervals
	fine_Ci = np.zeros(len(coarse_Ci))

	(A, B, l) = fine_basis.dfly(D, discrete=True)
	(AA, BB, ll) = fine_basis.dfly(D, discrete=True, n=1)
	M = fine_basis.bound_on_norms_of_powers(D)
	PM = coarse_basis.bound_on_norms_of_powers(D, project_left=True)
	Kdelta = coarse_basis.projection_error()
	Xi = fine_basis.strong_to_weak_norm_equivalence()

	for n, C in enumerate(coarse_Ci):
		# doing sums like this is quadratic in n, but we don't care.
		weak_coefficient = Kdelta * sum(coarse_Ci[i] for i in range(n)) * ((AA*ll+PM)*B + M*BB)
		strong_coefficient = Kdelta * sum(coarse_Ci[i]*(l**(n-i-1)) for i in range(n)) * (AA*ll+PM)
		error_bound = weak_coefficient + strong_coefficient * Xi
		fine_Ci[n] = (C + error_bound).magnitude()

	print("Last error bound: %f" % error_bound.magnitude())

	weak_fraction = weak_coefficient / error_bound
	if weak_fraction < 0.9:
		print("The error bound may be suboptimal -- consider extending Ci with extend_power_norms")

	for n in range(len(fine_Ci)):
		if fine_Ci[n] > M:
			fine_Ci[n] = M

	return fine_Ci

def power_norms_from_smaller_grid_alternate(D, coarse_basis, fine_basis, coarse_Ci, norm_fineP):
	"""
	Estimates power norms Ci for a finer basis using those for a coarser basis

	Uses methods from our notes matricine.pdf and integral_injection_filling.pdf

	Returns:
		fine_Ci: vector(RNDU) of estimates
	"""

	coarse_Ci = vector(D.field, coarse_Ci)
	
	m = len(coarse_Ci)

	fine_Ci = vector(D.field, m)

	(strong_bounds, ly_weak_bounds) = fine_basis.small_matrix_bounds(D, m)

	coefficient_of_Lipf = coarse_basis.normalization_norm() * coarse_basis.projection_weak_norm() * fine_basis.bound_on_operator_norm(D) * coarse_basis.projection_error()
	coefficient_of_LipLf = coarse_basis.normalization_norm() * coarse_basis.projection_error()

	fine_Ci[0] = 1
	for i in range(1,m):
		fine_Ci[i] = coarse_Ci[i] + sum(coarse_Ci[i-1-j]*(coefficient_of_Lipf*strong_bounds[j] + coefficient_of_LipLf*strong_bounds[j+1]) for j in range(i))

	return fine_Ci


def power_norms_table(D, coarse_basis, fine_basis, coarse_Ci):
	"""
	Builds the table mentioned in our internal notes file dubious_derivation.pdf
	"""

	(A, B, l) = fine_basis.dfly(D, discrete=True)

	n = len(coarse_Ci)
	T = np.zeros((n, n))
	for i in range(n):
		for j in range(i):
			T[i,j] = (coarse_Ci[j]*(l**(i-1-j))).magnitude()

	return T

def extend_power_norms(Ci, n=None):
	"""
	Computes automatically more/better bounds for the norms of ||A^k|| from the given ones

	Given a sequence such that Ci[i] \leq \|A^i\| (of arbitrary length),
	computes a better or longer sequence by using the relations 
	:math:`\|A^{i+j}\| \leq \|A^i\| \|A^j\|`.
	"""

	m = len(Ci)
	if n is None:
		n = m

	Ci = vector(RealField(rnd='RNDU'), Ci) 
	Di = vector(RealField(rnd='RNDU'), n)
	Di[0] = 1.
	Di[1] = Ci[1]

	for j in range(2, n):
		Di[j] = min(Di[i] * Di[j-i] for i in range(1, j))
		if j < m:
			Di[j] = min(Di[j], Ci[j])

	return Di


import unittest
from sage.all import RIF

class BasicTest(unittest.TestCase):
	"""
	Some tests.
	"""
	def test_extend(self):
		Ci = [1, 4, 0.125, 2, 4, 1, 0.25]
		assert extend_power_norms(Ci, 10) == (1.00000000000000, 4.00000000000000, 0.125000000000000, 0.500000000000000, 0.0156250000000000, 0.0625000000000000, 0.00195312500000000, 0.00781250000000000, 0.000244140625000000, 0.000976562500000000)


if __name__ == '__main__':
		unittest.main()
