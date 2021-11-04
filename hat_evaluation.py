"""
Finite element method based on hat functions + function evaluation as the dual basis.

For now, we throw in this file also the definition of the choices of an abstract finite element method (basis, norm)
"""

from partition import check_partition, partition_diameter, partition_minimum_diameter, nonzero_on, is_refinement
from basis import Basis
from dynamic import IterateDynamic, normalized
from sage.rings.real_mpfi import RealIntervalField, is_RealIntervalFieldElement

import numpy as np
from sage.all import load, RIF, RealNumber, RealField, vector, infinity, matrix, block_matrix, QQ
from sparse import max_nonzeros_per_row, matrix_norminf
from warnings import warn
from ulam import UlamL1

def evaluate_hat_function(lo, mi, hi, x):
	"""
	Evaluates a hat function in R --- no quotienting / normalization, just a regular hat function.

	The function is so that f(x) = 0 for x <= lo and x>= hi, f(mi)=1, 
	and piecewise linear in between.
	"""

	# It is easier to implement this properly if you see a hat function
	# defined as min / max of linear functions. 

	if not is_RealIntervalFieldElement(x):
		x = RealIntervalField()(x)

	left_branch = (x-lo)/(mi-lo)
	right_branch = (hi-x)/(hi-mi)
	return left_branch.min(right_branch).max(0).min(1)

def evaluate_double_hat(partition, i, x):
	"""
	Evaluates a function with two "hats" centered in `partition[i]` and `partition[i+1]`

	Assumes x normalized.

	This is the core of `HatFunctions.evaluate()`, but without the 
	quotientation / interval normalization so that we can more easily
	plot it to check that it is correct.

	Test with:
	import hat_evaluation
	import partition
	p = partition.equispaced(8)
	plot(lambda x: RR(hat_evaluation.evaluate_double_hat(p, 0, x)), xmin=0, xmax=2)
	"""

	if i < 0 or i >= len(partition) - 1:
		raise ValueError

	if not is_RealIntervalFieldElement(x):
		x = RealIntervalField()(x)

	field = x.parent()

	if i == 0:
		mi = field(partition[0])
		hi = field(partition[1])
		lo = field(partition[-2])
		first_hat = evaluate_hat_function(lo - 1, mi, hi, x)
		second_hat = evaluate_hat_function(lo, mi + 1, hi + 1, x)
		return first_hat.max(second_hat)

	else:
		lo = field(partition[i-1])
		mi = field(partition[i])
		hi = field(partition[i+1])

		first_hat = evaluate_hat_function(lo, mi, hi, x)
		second_hat = evaluate_hat_function(lo + 1, mi + 1, hi + 1, x)
		return first_hat.max(second_hat)

class HatFunctions(Basis):
	"""
	Abstract class for all bases composed of "hat functions"
	
	:math:`\phi_j` is the function so that :math:`\phi_j(x_i)=\delta_{ij}`, so :math:`\phi_0` has support :math:`[x_{n-1},1] \cup [0,x_1]`.
	
	"""
	def __init__(self, partition):
		check_partition(partition)
		self.partition = partition
	
	def __len__(self):
		return len(self.partition) - 1
	
	def evaluate(self, i, x):
		"""
		Evaluates a basis function on a point (may be non-normalized).
		"""
		x = normalized(x)

		# Since our normalized interval lives in [0, 2], there are actually two 'hats'
		# to keep track of. In the case i==0, there is a full hat and two half-hats
		# around 0 and 2, but since the lower bound is always within [0,1), every
		# time that we need the half-hat around 2 the result is already the full RIF(0, 1).

		return evaluate_double_hat(self.partition, i, x)

	def nonzero_on(self, I):

		# the basis function number `i` is nonzero on partition intervals `i` and `(i-1) % n`

		n = len(self)

		intervals = set(nonzero_on(I, self.partition))
		other_intervals = set((i+1) % n for i in intervals)

		return intervals.union(other_intervals)

	def evaluate_integral(self, i, ring=RIF):
		"""
		Integral of the `i`th basis function
		"""
		if i == 0:
			return (ring(self.partition[1])-ring(self.partition[0]) + ring(self.partition[-1])-ring(self.partition[-2])) / 2
		else:
			return (ring(self.partition[i+1]) - ring(self.partition[i-1])) / 2

class HatFunctionsWithEvaluations(HatFunctions):
	"""
	Hat functions + evaluation functions :math:`V_i(f) = f(x_i)` as dual basis.
	"""
	
	def __init__(self, partition):
		HatFunctions.__init__(self, partition)
	
	def dual_composed_with_dynamic(self, dynamic, epsilon):
		"""
		For each `i`, yields (successively) the sequence :math:`\{(i,y,|T'(y)|) \colon y\in T^{-1}(x_i)\}`.
		"""
		for i, x in enumerate(self.partition[:-1]):
			for y in dynamic.preimages(x, epsilon):
				yield (i, (y, (dynamic.fprime(y)).abs() ) )
		
	def project_dual_element(self, dual_element):
		y, absTprimey = dual_element
		for j in self.nonzero_on(y):
			yield (j, self.evaluate(j,y) / absTprimey)

	def sanitize_perron_vector(self, v):
		"""
		Uses theoretical knowledge on the Perron vector (e.g., realness, positivity...) to correct numerical output.
		
		This gets called only before computing the rigorous residual, so don't worry, we're not cheating.
		
		Args:
			v (numpy vector):
		Returns:
			v (numpy vector): the same vector but "adjusted". In particular, we require that it is scaled so that the invariant measure is positive and has mass 1 (numerically)
		"""
		
		v = np.real(v)
		v = v / sum(v) # makes sure that the components are (almost all) positive
		v[v<0] = 0
		v = v / self.integral(vector(v))
		return v

class HatFunctionsWithEvaluationsLinf(HatFunctionsWithEvaluations):
	def invariant_measure_strong_norm_bound(self, dynamic):

		# note that the "generic" bound in basis.py does not hold, since ||f||_inf is larger than 1.
		raise NotImplementedError

	def contracting_pairs(self):
		"""
		Return vectors of the form (1,0,0,...,0,-1,0,0,...) and 1/n.
		"""
		n = len(self)
		for i in range(n-1):
			u = np.zeros(n)
			u[0] = 1.
			u[i+1] = -1.
			yield (u, float(RealNumber(1,rnd='RNDD')/n))

	def norm_estimate(self, v):
		# Numpy's inf-norm should be exact, it's just abs() and max()
		return RealNumber(np.linalg.norm(v, np.inf), rnd='RNDU')

	def matrix_norm_estimate(self, PP):
		return matrix_norminf(PP, rnd='RNDU')

	def rigorous_norm(self, v):
		# care here: there is a Sage bug, v.norm(Infinity) does not work (http://trac.sagemath.org/ticket/17728).
		# and nor does Python's max(...)
		nm = v.base_ring().zero()
		for x in v:
			nm = nm.max(x.abs())
		return nm


	def projection_strong_norm(self):
		"""
		bound on Lip(Pf) / Lip(f)
		"""
		# Follows from Lip Pf = Lip f + 1/alpha*|int f - uVf|*Lip(h), using the notation in nonzerointegral.pdf
		Rup = RealField(rnd='RNDU')
		return Rup(1)


	def projection_weak_norm(self):
		Rup = RealField(rnd='RNDU')
		return Rup(1)


	def bound_on_norms_of_powers(self, dynamic, project_left = False, project_right = False):
		# This should hold also for the projected version, since |Pi| = 1 in our case.

		# this gives a tighter bound, usually
		if isinstance(dynamic, IterateDynamic):
			dynamic = dynamic._D

		# Bound straight from Bahsoun-Bose 2010, Sec. 3.1

		# We use a UlamL1 basis to extract the coefficients of the Var-L1 Lasota-Yorke.
		# The fact that we need to supply a dummy partition probably means
		# that our type system is a bit borked.
		ulambasis = UlamL1([0., 1.])
		(A, B, lam) = ulambasis.dfly(dynamic)

		return (B + 1).magnitude()

	def bound_on_operator_norm(self, dynamic):
		"""
		Returns a bound on ||L||_w
		"""
		return self.bound_on_norms_of_powers(dynamic)

	def matrix_norm_diameter(self, P):
		w = vector(RealField(rnd='RNDU'),P.dimensions()[0])
		for (ij, val) in P.dict().iteritems():
			w[ij[0]] += val.absolute_diameter() #operations are rounded up because the LHS is so
		return max(w)

	def numerical_error(self, dynamic, P, PP):
		# Note that this is the exact same implementation as in ulam.py
		Rup = RealField(rnd='RNDU')
		Rdown = RealField(rnd='RNDD')
		
		D = self.matrix_norm_diameter(P)
		
		gamma = Rup(np.finfo(float).eps) * max_nonzeros_per_row(P);
		gamma = gamma / (Rdown(1)-gamma)
		
		K = gamma * self.matrix_norm_estimate(PP) + D
		return K

	def lasota_yorke_constants(self, dynamic):
		warn('Deprecated: we are trying to move to dfly')

		M = self.bound_on_norms_of_powers(dynamic)
		alpha = M / dynamic.expansivity()
		if not alpha < 1:
			raise ValueError("This map does not admit a Lasota-Yorke. Try with an iterate (see :func:`iterate_with_lasota_yorke`)")
		# from [Galatolo, Nisoli, Lemma 17 with n=1]
		return (alpha, M*(1+dynamic.lipschitz_of_L1() / (1-alpha))) 

	def semidiscrete_lasota_yorke_constants(self, dynamic):
		# the projection operators are contractions, so everything works fine
		return self.lasota_yorke_constants(dynamic)

	def mixed_matrix_norm_approximation_error(self, dynamic):
		return partition_diameter(self.partition) * (self.bound_on_norms_of_powers(dynamic) + 1)

	def mixed_matrix_norm_distance(self, other):

		# TODO: this is essentially the bound delta * Lip. Can we do better, like delta/2*Lip?
		if not type(other) == type(self):
			raise NotImplementedError

		fine = self.partition
		coarse = other.partition
		if len(fine) < len(coarse):
			fine, coarse = coarse, fine

		if is_refinement(fine, coarse):
			return partition_diameter(coarse)
		else:
			return partition_diameter(coarse) + partition_diameter(fine)

	def strong_to_weak_norm_equivalence(self):
		"""
		Return the constant required to estimate a strong norm with a weak norm (in the discretized space)

		Since we have a finite-dimensional discretized space, there is a constant
		such that :math:`\|f\|_s \leq C \|f\|_w` for each `f` in the discrete space.

		Returns:
			C (real with RNDU):
		"""
		
		return RealField(rnd='RNDU')(2) / partition_minimum_diameter(self.partition)

	def iterate_with_lasota_yorke(self, D):
		warn('Deprecated: we are trying to move to dfly')

		l = 1 / D.expansivity()
		M = self.bound_on_norms_of_powers(D)
		# return self if it works
		if M*l < 1:
			return D
		if not l < 1:
			raise ValueError("No iterate of this map admits a Lasota-Yorke")
		# we compute k in this way to avoid approximation errors. If k is very large, things are probably going to be slow later anyways
		k = 2
		while not M*(l **k) < 1:
			k = k + 1
		return IterateDynamic(D, k)

	def dfly(self, dynamic, discrete=False, n=None):
		"""
		Return constants of the dfly inequality :math:`\operatorname{Lip} L^n f \leq A \lambda^n \operatorname{Lip} f + B\|f\|_w` 

		This function should make `lasota_yorke_constants` and `iterate_with_lasota_yorke` obsolete.
		
		Input:
			Dynamic:
			discrete: if True, returns constants for the projected operator instead

		Returns:
			(A, B, lambda): tuple of intervals
		"""

		warn('Double-check this, probably it can be improved')
		M = self.bound_on_norms_of_powers(dynamic)
		alpha = M / dynamic.expansivity()
		if not alpha < 1:
			raise NotImplementedError('There is probably a way to make a DFLY for this map but we need to check')
		# from [Galatolo, Nisoli, Lemma 17]
		if n is None:
			return (dynamic.field(1), M*(1+dynamic.lipschitz_of_L1() / (1-alpha)), alpha)
		elif n==1:
			return (dynamic.field(1), dynamic.lipschitz_of_L1(), alpha)			
		else:
			# if n is specified, we can truncate the sum 1+alpha+alpha^2+... earlier
			return (dynamic.field(1), M*(dynamic.lipschitz_of_L1() * (1-alpha**n) / (1-alpha)), alpha)

	def projection_error(self):
		r"""
		Returns a bound on the projection error :math:`\|\Pi-I\|_{s\to w}`
		"""
		return partition_diameter(self.partition) / 2

	def extension_operator(self, other):
		"""
		Extension operator from a vector of length nc to one of length nf

		# TODO: implemented only for equispaced grids
		"""
		if not type(other) == type(self):
			raise NotImplementedError

		fine = self
		coarse = other
		if len(fine) < len(coarse):
			fine, coarse = coarse, fine

		if not is_refinement(fine.partition, coarse.partition):
			raise NotImplementedError

		nf = len(fine)
		nc = len(coarse)

		m = nf / nc

		E = matrix(QQ, nf, nc, sparse=True)
		M = matrix([range(m, 0, -1)], ring=QQ).transpose() / m
		for i in range(nc):
			E[i*m:i*m+m, i] = M
			E[i*m+1:i*m+m, (i+1)%nc] = M[:0:-1, 0]

		return E

	def restriction_operator(self, other):
		"""
		Restriction operator from a vector of length nf to one of length nc

		# TODO: implemented only for equispaced grids
		"""
		if not type(other) == type(self):
			raise NotImplementedError

		fine = self
		coarse = other
		if len(fine) < len(coarse):
			fine, coarse = coarse, fine

		if not is_refinement(fine.partition, coarse.partition):
			raise NotImplementedError

		nf = len(fine)
		nc = len(coarse)

		m = nf / nc

		R = matrix(QQ, nc, nf, sparse=True)
		for i in range(nc):
			R[i, m*i] = 1

		return R

import unittest

class BasicTest(unittest.TestCase):
	"""
	Some tests.
	"""
	def test_evaluate(self):
		import partition
		from sage.all import RIF

		assert evaluate_hat_function(0, 1, 2, 1) == 1.
		assert evaluate_hat_function(0, 1, 2, 1.5) == 0.5
		assert evaluate_hat_function(0, 1, 2, 2.5) == 0
		assert evaluate_hat_function(0, 1, 2, RIF(-1,3)).endpoints() == RIF(0,1).endpoints()
		assert evaluate_hat_function(0, 1, 2, RIF(-0.5,0.5)).endpoints() == RIF(0,0.5).endpoints()
		assert evaluate_hat_function(0, 1, 2, RIF(1.5,2.5)).endpoints() == RIF(0,0.5).endpoints()

		basis = HatFunctionsWithEvaluationsLinf(p)
		assert basis.evaluate(0, 0) == 1
		assert basis.evaluate(1, 0) == 0
		assert basis.evaluate(0, RIF(0.875, 1.125)).endpoints() == (0.5, 1)
		assert basis.evaluate(1, RIF(0.875, 1.125)).endpoints() == (0, 0.5)
		assert basis.evaluate(2, RIF(0.875, 1.8)).endpoints() == (0, 1)

	def test_nonzero_on(self):
		import partition
		from sage.all import RIF
		p = partition.equispaced(4)
		basis = HatFunctionsWithEvaluationsLinf(p)

		def nonzero_on2(x):
			# note that this is slightly different from basis.nonzero_on(), because
			# basis.nonzero_on will return a partition interval if it intersects
			# x in a point, while this will not.
			# We solve the problem by never testing it on interval extrema.
			return set(i for i in range(len(basis)) if not basis.evaluate(i, x) == 0.)

		assert set(basis.nonzero_on(RIF(0.01,0.2))) == set(nonzero_on2(RIF(0.01,0.2)))
		assert set(basis.nonzero_on(RIF(0.4,0.9))) == set(nonzero_on2(RIF(0.4,0.9)))
		assert set(basis.nonzero_on(RIF(0.9,1.1))) == set(nonzero_on2(RIF(0.9,1.1)))
		assert set(basis.nonzero_on(RIF(0.8,1.7))) == set(nonzero_on2(RIF(0.8,1.7)))
		assert set(basis.nonzero_on(RIF(0.8,1.4))) == set(nonzero_on2(RIF(0.8,1.4)))

if __name__ == '__main__':
		unittest.main()
