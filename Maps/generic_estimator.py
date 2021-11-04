"""
Norm-agnostic functions for getting error estimates
"""

from itertools import count

def error_bound_from_decay_time(D, basis, n, alpha, residual):
	"""
	Estimates the error on the computed invariant measure using the decay time
	"""
	
	M = basis.bound_on_norms_of_powers(D, project_left=True, project_right=True)
	
	if not alpha < 1:
		raise ValueError, 'alpha has to be smaller than 1 (true contraction)'
	
	amplification_factor = M * n / (D.field(1)-alpha)
	
	discretization_error = basis.invariant_measure_strong_norm_bound(D) * basis.mixed_matrix_norm_approximation_error(D)

	error = (discretization_error + residual) * amplification_factor
	return error.upper()

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
			raise ValueError, "Insufficient decay time estimate. Retry with a finer partition or a smaller coarse_alpha."

		mink = first_below(f, 1)
		maxk = max(100,5*mink) #crude ballpark heuristic TODO: can we do better?
		objective_function = lambda k: (coarse_n + k) / (1 - f(k).center())
		k = minimize_on_integers(objective_function, mink, maxk)
		return coarse_n + k, f(k).magnitude()
	else:
		if not C1 < fine_alpha:
			raise ValueError, "Insufficient decay time estimate. Retry with a finer partition or a smaller coarse_alpha."
		k = first_below(f, fine_alpha)
		return coarse_n + k, fine_alpha
