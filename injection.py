
"""
Functions pertaining the "integral injection" trick for the Linf basis
"""

from sage.all import RealField
from hat_evaluation import *
import numpy as np
from sparse import norm1, gamma, max_nonzeros_per_row, matrix_norm1, matrix_norminf, sage_sparse_to_scipy_sparse, norminf
from sage.all import VectorSpace, RR, vector


def evaluate_sawtooth(partition, x):
	"""
	Evaluates the "sawtooth function" which is 0 on the points of a partition and 1 on their midpoints.

	Assumes x normalized.

	Note that these midpoints are *not* computed rigorously, since it is not necessary: a function that
	is 1 on any point between the partition points works in the same way for what we are trying to do.
	"""

	if not is_RealIntervalFieldElement(x):
		x = RealIntervalField()(x)

	field = x.parent()

	intervals = nonzero_on(x, partition)
	y = field(-infinity)

	for i in intervals:
		lo = field(partition[i])
		hi = field(partition[i+1])
		mi = field((lo + hi) / 2)

		first_hat = evaluate_hat_function(lo, mi, hi, x)
		second_hat = evaluate_hat_function(lo+1, mi+1, hi+1, x)
		hat = first_hat.max(second_hat)
		y = y.max(hat)

	return y

class HatFunctionsInjected(Basis):
	"""
	Extends the HatFunctions basis with an "integral injection".

	The last basis function is the piecewise linear function that
	is zero on the points of self.partition and `2*integral_value` on their midpoints 
	(hence it has integral `integral_value`).
	"""

	def __init__(self, partition, integral_value):
		check_partition(partition)
		self.partition = partition
		self.integral_value = integral_value
		self.original_basis = HatFunctionsWithEvaluationsLinf(partition)

	def __len__(self):
		return len(self.partition) - 1 + 1

	def evaluate(self, i, x):
		if i == len(self) - 1:
			return evaluate_sawtooth(self.partition, x)
		else:
			return self.original_basis.evaluate(i, x)

class HatFunctionsInjectedWithEvaluations(HatFunctionsInjected):
	"""
	The last dual function is 1/integral_value*(int_X f - sum_{x_i in partition} f(x_i) * self.evaluate_integral(i))
	"""

	def __init__(self, partition, integral_value):
		HatFunctionsInjected.__init__(self, partition, integral_value)

	def assemble(self, dynamic, epsilon, prec=53):
		n = len(self.partition) - 1

		P = self.original_basis.assemble(dynamic, epsilon, prec)

		last_column = matrix(P.base_ring(), n, 1, sparse=True)
		for i, dual_element in self.original_basis.dual_composed_with_dynamic(dynamic, epsilon):
			y, absTprimey = dual_element
			last_column[i, 0] += evaluate_sawtooth(self.partition, y) / absTprimey

		last_column = last_column * self.integral_value * 2 # corrects the fact that we used a sawtooth function of height 1, so with integral 1/2

		# must be sparse, otherwise trapezoid_method * P below tries to convert P to full. :(
		trapezoid_method = matrix(P.base_ring(), 1, n, (self.original_basis.evaluate_integral(i) for i in range(n)), sparse=True)

		last_row = (trapezoid_method * P).apply_map(lambda x: self.original_basis.evaluate_integral(i)-x) / self.integral_value
		last_entry = (self.integral_value - trapezoid_method * last_column) / self.integral_value

		return block_matrix([[P, last_column], [last_row, last_entry]], sparse=True)

	def sanitize_perron_vector(self, v):
		# we know that the eigenvector can be chosen real, with nonnegative entries apart from maybe the last one, and with integral 1
		v = np.real(v)
		v = v / sum(v[0:-1]) # makes sure that the components are (almost all) positive
		n = len(self.partition) - 1
		for i in range(n):
			if v[i] < 0:
				v[i] = 0
		v = v / self.integral(vector(v))  # produces a function with integral 1
		return v

	def evaluate_integral(self, i, ring=RIF):
		if i == len(self) - 1:
			return self.integral_value
		else:
			return self.original_basis.evaluate_integral(i, ring)

class HatFunctionsInjectedWithEvaluationsLinf(HatFunctionsInjectedWithEvaluations):
	"""
	We use the Linf norm in both the continuous setting and the discrete setting.
	"""

	def norm_estimate(self, v):
		# Numpy's inf-norm should be exact, it's just abs() and max()	
		Rup = RealField(rnd='RNDU')
		return Rup(np.linalg.norm(v[0:-1], np.inf)) + Rup(2)*self.integral_value*v[-1]

	def lasota_yorke_constants(self, dynamic):
		return self.original_basis.lasota_yorke_constants(dynamic)


	def projection_strong_norm(self):
		# Follows from Lip Pf = Lip f + 1/alpha*|int f - uVf|*Lip(h), using the notation in nonzerointegral.pdf
		Rup = RealField(rnd='RNDU')
		return Rup(2.)


	def projection_weak_norm(self):
		Rup = RealField(rnd='RNDU')
		return Rup(3.)


	def mixed_matrix_norm_approximation_error(self, dynamic):
		warn('TODO: the estimate in modified Theorem 1 can probably be improved to require only one projection in most of the places, so saving a factor ||Pi||')
		return self.projection_error() * (self.projection_weak_norm() * self.bound_on_norms_of_powers(dynamic) + 1)


	def dfly(self, dynamic, discrete=False, n=None):
		if discrete:
			(A, B, l) = self.original_basis.dfly(dynamic, discrete, 1)
			Ps = self.projection_strong_norm()
			Pw = self.projection_weak_norm()
			alpha = Ps * l
			if not alpha < 1:
				raise NotImplementedError
			if not A == 1:
				raise NotImplementedError
			warn('Double-check this bound!!')
			if n is None:
				return (A, B*Pw / (1-alpha), alpha) 
			else:
				return (A, B*Pw * (1-alpha**n) / (1-alpha), alpha) 
		else:
			return self.original_basis.dfly(dynamic, discrete, n)


	def continuous_to_discrete_norm_equivalence(self):
		"""
		Constant :math:`\|U\|` such that :math:`\|Uv\| \leq C\|v\|` for all vectors :math:`v`.
		(see nonzerointegral.pdf)

		Returns:
			real with RNDU
		"""
		# TODO: I think we now use them only to build the norm of the projection, so we can get rid of it.
		Rup = RealField(rnd='RNDU')

		return Rup(1.)


	def discrete_to_continuous_norm_equivalence(self):
		"""
		Constant :math:`\|V\|` such that :math:`\|Vf\| \leq C\|f\|` for all functions :math:`f`.
		(see nonzerointegral.pdf)

		Returns:
			real with RNDU
		"""
		Rup = RealField(rnd='RNDU')

		return Rup(1) + Rup(2)*Rup(self.integral_value).abs()


	def bound_on_norms_of_powers(self, dynamic, project_left = False, project_right = False):

		# this gives a tighter bound for iterate maps, usually
		if isinstance(dynamic, IterateDynamic):
			dynamic = dynamic._D

		# Bound straight from Bahsoun-Bose 2010, Sec. 3.1

		# We use a UlamL1 basis to extract the coefficients of the Var-L1 Lasota-Yorke.
		# The fact that we need to supply a dummy partition probably means
		# that our type system is a bit borked.
		ulambasis = UlamL1([0., 1.])
		(A, B, lam) = ulambasis.dfly(dynamic)

		bound = (B + 1).magnitude()

		warn("TODO: double-check this bound??")
		# By definition, this uses Pi = U*V, so we cannot skip factors
		# TODO: double-check if both these products are needed
		normPi = self.discrete_to_continuous_norm_equivalence() * self.continuous_to_discrete_norm_equivalence()
		if project_left:
			bound = bound * normPi 
		if project_right:
			bound = bound * normPi

		return bound


	def bound_on_norms_of_matrix_powers(self, dynamic):
		"""
		Modified version of bound_on_norms_of_powers that keeps into account the split
		between norms on the continuous and discrete space (and returns a tighter bound than
		`bound_on_norms_of_powers(dynamic, True, True)`).

		This bounds :math:`\|P^k\|` uniformly in :math:`k`, where :math:`P` is the
		Ulam matrix.
		"""
		# TODO: this should be currently unused
		return self.discrete_to_continuous_norm_equivalence() * self.continuous_to_discrete_norm_equivalence() * \
			self.bound_on_norms_of_powers(dynamic)


	def projection_error(self):
		# This is twice as the projection error delta/2 for the non-injected function
		# Indeed, the coefficient in front of h can be as large as 1/alpha*delta/4*Lip f (see nonzerointegral.pdf),
		# and h has inf-norm 2alpha, so we get an additional summand delta/2.
		return partition_diameter(self.partition)


	def strong_to_weak_norm_equivalence(self):
		"""
		Return the constant required to estimate a strong norm with a weak norm (in the discretized space)

		Since we have a finite-dimensional discretized space, there is a constant
		such that :math:`\|f\|_s \leq C \|f\|_w` for each `f` in the discrete space.

		Returns:
			C (real with RNDU):
		"""
		# A function in the discretized injected space is piecewise linear with "pieces" large delta/2.
		# So the worst case is when we change from -1 to 1 on one of those pieces.
		return RealField(rnd='RNDU')(4) / partition_minimum_diameter(self.partition)


def norms_of_powers(P, M, integral_value, m=20, K=RealField(rnd='RNDU').zero()):
	"""
	Estimates :math:`\| A^k |V \|` for k in range(m)

	:math:`A` is an operator such that :math:`AV \subseteq V`
	:math:`|V` means that the norm is restricted to the set of vectors orthogonal to [1/n 1/n ... 1/n alpha]
	(the last entry is different).

	If one uses equispaced nodes and "injects" a function with integral alpha, then the last entry is
	1/alpha*(int-uV), and int f = [1/n 1/n ... 1/n alpha]*\hat{V}*f. Hence a basis of the space of
	functions with integral 0 is [0 0 ... 0 1 0 ... 0 0 -1/(alpha*n)].

	Args:
		P (scipy matrix): a matrix such that :math:`\|P-A\| \leq K`
		integral_value (positive real): see above
		m (integer): how many norms to compute
		M, K (reals with RNDU): constants such that :math:`\|P-A\| \leq K` and :math:`\|A^i\| \leq M` for each i in `range(m)`.
	
	Returns:
		a numpy array of norm bounds, starting from `k=0` (which is simply 1)
    """

	n = P.shape[1] - 1
	prop = (gamma(max_nonzeros_per_row(P)) * matrix_norminf(P, rnd='RNDU') + K) * M
	zero = RealField(rnd='RNDU').zero()

	eta = 1/(RIF(integral_value) * RIF(n))
	if not eta.is_exact():
		raise NotImplementedError("This method only works if 1/(n*eta) can be computed exactly in floating point -- otherwise we need some minor changes")
	eta = RealField(rnd='RNDU')(eta)

	sums = np.zeros((n + 1, m))
	for j in range(0, n):
		v = np.zeros(n+1)
		v[j] = 1.
		v[n] = -eta

		current_norm = eta + 1.

		error = zero
		for i in range(1,m):
			v = P * v
			error += prop * current_norm
			current_norm = norminf(v, rnd='RNDU') #TODO: not sure norminf is correct here?
			sums[:,i] = sums[:,i] + abs(v) + error

	Ci = np.zeros(m)
	Ci[0] = 1.
	Rup = RealField(rnd='RNDU')
	for i in range(1, m):
		# the term 1+gamma(2*i) keeps track of the errors obtained when summing
		# sums[:,i] + v + error (summation error on each component)
		# TODO: if proper rounding is used, this term could be omitted.
		# Not that it matters much, though.
		Ci[i] = Rup(max(sums[0:n, i]))*(Rup(1)+gamma(2*i)) + Rup(2)*integral_value*sums[n, i]*(Rup(1)+gamma(2*i))

	for i in range(1, m): #if the bounds are larger than M, replace them with M
		Ci[i] = min(Ci[i], M)

	return Ci

def interval_norm_error(P, integral_value):
	"""
	Compute the "norm width" of the interval matrix P, i.e., an upper bound for :math:`\|P_1-P_2\|`, where the :math:`P_i` are matrices inside the interval :math:`P`,
	
	The norm here is the "injected norm" 
	"""
	n = P.dimensions()[0]
	w = VectorSpace(RealField(rnd='RNDU'),n)(0)
	for (ij, val) in P.dict().iteritems():
		w[ij[0]] += val.absolute_diameter() #operations are rounded up because the LHS is so
	return max(w[0:-1]) + w[-1]*2*integral_value


class BasicTest(unittest.TestCase):
	"""
	Some tests.
	"""
	def test_evaluate_sawtooth(self):
		import partition
		from sage.all import RIF

		p = partition.equispaced(4)
		assert evaluate_sawtooth(p, 0.125) == 1
		assert evaluate_sawtooth(p, 0.25) == 0
		assert evaluate_sawtooth(p, RIF(0, 0.125)).endpoints() == (0, 1)
		assert evaluate_sawtooth(p, RIF(0.90625,1)).endpoints() == (0, 0.75)

if __name__ == '__main__':
		unittest.main()
