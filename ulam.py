"""
Ulam basis written in the generic assembler framework
"""

from __future__ import division
from basis import Basis
from partition import check_partition, partition_diameter, partition_minimum_diameter, is_refinement, is_inside, overlap, nonzero_on

from sage.all import load, RealNumber, RIF

import numpy as np
from sage.all import vector, RealField, RealIntervalField
from sparse import max_nonzeros_per_row, norm1, matrix_norm1, sparse_matvec
from warnings import warn
from interval import interval_newton, NoZeroException

from sage.rings.real_mpfi import RealIntervalField, is_RealIntervalFieldElement
from dynamic import normalized, Mod1Dynamic, QuotientedLinearDynamic, PiecewiseExpandingDynamic
load('binsearch2.spyx')

class Ulam(Basis):
	r"""
	Basis for the Ulam method
	
	Basis eleme`ts are the functions :math:`u_j = 1_{[x_j,x_{j+1}]}`
	
	Dual basis elements are the integrals :math:`V_i = \frac{1}{|x_{i+1}-x_i|} \int_{[x_i,x_{i+1}]}`.
	
	Note that we need some care with how the normalization interacts with non-equispaced intervals:
	If the partition is e.g. [0,1/4,1/2,1]:
	* the identically-1 function is mapped to the vector V1 = [1/4;1/4;1/2]
	* the integral is mapped to the row vector u^T = [1/4;1/4;1/2]
	* the vector norm ||w|| = ||Uw|| is the usual |v_1|+|v_2|+|v_3|.
	
	In particular, the matrix P is not stochastic anymore, but satisfies u^T*P = u^T.
	"""
	
	def __init__(self, partition):
		check_partition(partition)
		self.partition = partition
	
	def __len__(self):
		return len(self.partition) - 1
	
	def evaluate(self, i, x):
		# updated to support canonical representatives
		field = x.parent()

		if is_inside(x, self.partition, i):
			return field(1)
		elif overlap(x, self.partition, i):
			return field(0, 1)
		else:
			return field(0)
		
	def nonzero_on(self, I):
		return nonzero_on(I, self.partition)
			
	def dual_composed_with_dynamic(self, dynamic, epsilon):
		r"""
		For each `i`, yields the set of `dynamic.nbranches` pairs :math:`\{(f_k^{-1}(x_i),f_k^{-1}(x_{i+1})) \colon f_k \in \text{branch inverses}\}`.
		Note that this is a *pair* and not an *interval*. This is necessary, because we have
		uncertainty on the values of the lower and upper bound of the interval, and
		interval arithmetic is used to represent that. If we were to "flatten" the interval,
		we'd lose the interpretation of its minimum and maximum length.

		Repeat after me: interval arithmetic is *not* for representing Ulam intervals.
		"""

		if isinstance(dynamic, Mod1Dynamic):

			# We rely on the fact that the preimages are returned in increasing order here,
			# so be careful if you wish to implement this for IterateDynamic() as well.

			# TODO: this would probably need to be done in a completely different way
			# to be more general, for instance using the multi-interval generalization
			# of interval_newton.
			# It is probably going to be simpler to deal with these issues when we
			# generalize this thing to domains different from [0,1].

			n = len(self)
			preimages_0 = list(dynamic.preimages(self.partition[0], epsilon))
			preimages_xi = preimages_0
			for i in range(len(self) - 1):
				preimages_xi_plus_one = list(dynamic.preimages(self.partition[i+1], epsilon))
				for K in zip(preimages_xi, preimages_xi_plus_one):
					yield (i, K)
				preimages_xi = preimages_xi_plus_one

			# The last interval is special because we need the preimages of "1 unquotiented", not 0, 
			# and simply evaluating f would quotient incorrectly and give again preimages_0.
			preimages_1 = preimages_0[1:]
			preimages_1.append(dynamic.field(1))
			for K in zip(preimages_xi, preimages_1):
				yield (n-1, K)

		elif isinstance(dynamic, PiecewiseExpandingDynamic):
			for k in range(dynamic.nbranches):
				f = dynamic.functions[k]
				fprime = dynamic.derivatives[k]
				if not fprime((dynamic.grid[k]+dynamic.grid[k+1])/2) > 0:
					raise NotImplementedError('Only increasing functions for now')
				fmin = f(dynamic.field(dynamic.grid[k]))
				fmax = f(dynamic.field(dynamic.grid[k+1]))
				fdomain = dynamic.field(dynamic.grid[k], dynamic.grid[k+1])
				frange = fmin.union(fmax)
				# integer_shift must be an interval because we need to make arithmetic with it
				integer_shift = dynamic.field(fmin.floor().lower())
				lowest_i = binsearch2((fmin - integer_shift).lower(), self.partition)
				assert lowest_i > -1
				a = dynamic.field(dynamic.grid[k])
				i = lowest_i
				while True:
					next_i = i + 1
					if next_i == len(self):
						next_i = 0
						integer_shift = integer_shift + 1
					try:
						b =	interval_newton(f, fprime, fdomain, integer_shift + self.partition[next_i], epsilon)
					except NoZeroException:
						yield (i, (a, dynamic.field(dynamic.grid[k+1])))
						break
					yield (i, (a, b))
					i = next_i
					a = b
		else:
			raise NotImplementedError('We need guaranteed monotonicity intervals in the dynamic for this to work')

	def project_dual_element(self, dual_element):
		a, b = dual_element
		for j in self.nonzero_on(a.union(b)):
			# ensure that they are sorted
			A = a.min(b)
			B = a.max(b)
			
			x = A.parent()(self.partition[j])
			y = A.parent()(self.partition[j+1])
			
			# compute endpoints of the intersection
			lower = A.max(x)
			upper = B.min(y)
			yield (j, (upper - lower).max(0) / (y-x))

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
		v = v / np.linalg.norm(v, ord=1) #probably faster than our norm1(v), and than self.integral(v)
		return v

	def evaluate_integral(self, i, ring=RIF):
		"""
		Integral of the `i`th basis function
		"""
		return ring(1)

	def project_function_approximate(self, f, nsamples = 32):
		"""
		Returns an approximate 'projection' of a given function f on the Ulam basis, i.e., a step function obtained by discretization.
		"""
		def approximate_integral(f, a, b):
			"""
			Crap method, just to have something. That's not even the correct trapezoid method.
			"""
			return np.mean([f(x) for x in np.linspace(a, b, nsamples)])

		values = np.array([approximate_integral(f, self.partition[i], self.partition[i+1]) for i in range(len(self))])

		def g(x, values=values, partition=self.partition):
			"""
			Piecewise linear function that evaluates the projection
			"""
			if x<0 or x>1:
				raise ValueError('value out of range')
			bin = binsearch2(x, partition)
			return values[bin]

		return g

class UlamL1(Ulam):
	"""
	Combines Ulam with norm-1 bound estimation.
	"""
	
	def contracting_pairs(self):
		"""		
		Return vectors with u_i[0]=u_i[i+1]=1/2 and the rest 0, and s_i=1/2.
		"""
		n = len(self)
		
		for i in range(n-1):
			u = np.zeros(n)
			u[0] = 1/2
			u[i+1] = -1/2
			yield (u, 1/2)
		
	def norm_estimate(self, v):
		return norm1(v, rnd='RNDU')
	
	def matrix_norm_estimate(self, PP):
		return matrix_norm1(PP, rnd='RNDU')
	
	def rigorous_norm(self, v):
		return v.norm(1)
	
	def bound_on_norms_of_powers(self, dynamic, project_left = False, project_right = False):
		return RealField(rnd='RNDU')(1)
			
	def matrix_norm_diameter(self, P):		
		w = vector(RealField(rnd='RNDU'),P.dimensions()[1])
		for (ij, val) in P.dict().iteritems():
			w[ij[1]] += val.absolute_diameter() #operations are rounded up because the LHS is so
		return max(w)
	
	def numerical_error(self, dynamic, P, PP):
		r"""
		Return a bound for the error amplification coefficients.
		
		Args:
			dynamic
			P (sage interval matrix): the computed discretized matrix
			PP (scipy sparse matrix): the matrix that we will use for the floating-point arithmetic
		
		Returns:
			K (float): a constant such that :math:`\|Fl(Pv)-Pv\| \leq K\|v\|` for all vectors :math:`v`
		"""
		Rup = RealField(rnd='RNDU')
		Rdown = RealField(rnd='RNDD')
		
		D = self.matrix_norm_diameter(P)
		
		gamma = Rup(np.finfo(float).eps) * max_nonzeros_per_row(P);
		gamma = gamma / (Rdown(1)-gamma)
		
		K = gamma * self.matrix_norm_estimate(PP) + D
		return K

	def lasota_yorke_constants(self, dynamic):
		r"""
		Return Lasota-Yorke constants
		
		This is meant to replace `dynamic.lasota_yorke_constants()` with a more norm-agnostic packaging
		
		Returns:
			(lambda, B) (pair of real intervals): such that :math:`\|Lf\|_s \leq \lambda \|f\|_s + B\|f\|`
		"""
		warn('Deprecated: we are trying to move to dfly')

		if not dynamic.expansivity() > 1:
			raise ValueError('Your map is not expansive. A Lasota-Yorke inequality probably doesn''t exist.')

		if dynamic.is_continuous_on_torus():
			return (1/dynamic.expansivity(), dynamic.distorsion())
		else:
			if dynamic.expansivity() > 2:
				raise NotImplementedError('since this is deprecated, not implementing it')
			else:
				raise ValueError("We don't know how to make a LY inequality if lambda > 1/2. Try with 'iteratewithlasotayorke'.")
			raise NotImplementedError("L-Y constants are implemented only for functions which are continuous on the torus. Define a method is_continuous_on_torus() if your map is so.")

	def semidiscrete_lasota_yorke_constants(self, dynamic):
		r"""
		Return Lasota-Yorke constants of :math:`L\Pi`

		Returns:
			(lambda, B) (pair of real intervals): such that :math:`\|L\Pi f\|_s \leq \lambda \|f\|_s + B\|f\|`
		"""
		return self.lasota_yorke_constants(dynamic)

	def mixed_matrix_norm_approximation_error(self, dynamic):
		r"""
		Return a bound on :math:`\|L-L_h\|_{s \to w}`

		That is, :math:`\sup \frac{\|(L-L_h)f\|_w}{\|f\|_s}`, :math:`\|\cdot\|_w` and :math:`\|\cdot\|_s` 
		being the weak and strong norm, respectively.
		"""

		return partition_diameter(self.partition)*2

	def mixed_matrix_norm_distance(self, other):
		r"""
		Distance (in strong->weak norm) between the projection associated to this basis and to another one.

		Args:
			other: another basis
		Returns:
			a bound to :math:`\|\Pi_{self} - \Pi_{other}\|_{s\to w}`, where :math:`\Pi` denotes the projection operator
			If one partition is a refinement of the other, returns delta_C, else delta_C + delta_F.
		"""

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
		r"""
		Return the constant required to estimate a strong norm with a weak norm (in the discretized space)

		Since we have a finite-dimensional discretized space, there is a constant
		such that :math:`\|f\|_s \leq C \|f\|_w` for each `f` in the discrete space.

		Returns:
			C (real with RNDU):
		"""
		
		return RealField(rnd='RNDU')(2) / partition_minimum_diameter(self.partition)
		
	def iterate_with_lasota_yorke(self, D):
		if not D.is_continuous_on_torus():
			raise NotImplementedError
		if D.expansivity() > 1:
			return D
		else:
			raise ValueError("No iterate of this map admits a Lasota-Yorke inequality")


	def dfly(self, dynamic, discrete=False, n=None):
		r"""
		Return constants of the dfly inequality :math:`\|L^n f\|_s \leq A \lambda^n \|f\|_s + B\|f\|_w` 

		This function should make `lasota_yorke_constants` and `iterate_with_lasota_yorke` obsolete.
		
		Input:
			Dynamic:
			discrete: if True, returns constants for the projected operator instead
			n: if non-None, may return a (possibly stronger) inequality that holds only for
			   the given value of n

		Returns:
			(A, B, lambda): tuple of intervals
		"""
		if not dynamic.expansivity() > 1:
			raise ValueError('Your map is not expansive. A Lasota-Yorke inequality probably doesn''t exist.')

		if dynamic.is_continuous_on_torus():
			# note that this bound is ok for both continuous and discrete, since all projections have norm 1.
			lam = 1/dynamic.expansivity()
			if n==1:
				return (dynamic.field(1), dynamic.distorsion(), lam, )
			else:
				return (dynamic.field(1), dynamic.distorsion() / (1 - lam), lam, )
		elif isinstance(dynamic, QuotientedLinearDynamic):
			# Galatolo-Nisoli, Remark 8 with d_0=0, d_1=1. TODO: check with Isaia.
			lam = 1/dynamic.expansivity()
			if not 2*lam < 1:
				raise ValueError('Try with iterate_with_lasota_yorke(); maybe there is a sharper bound though')
			if n==1:
				return (dynamic.field(1), 2*dynamic.distorsion() + 2, 2*lam, )
			else:
				return (dynamic.field(1), (2*dynamic.distorsion() + 2) / (1-2*lam), 2*lam, ) 
		elif isinstance(dynamic, PiecewiseExpandingDynamic):
			# Galatolo-Nisoli, Remark 8 with d_0=0, d_1=1. 
			lam = 1/dynamic.expansivity()
			t = dynamic.field(0)
			for i in range(dynamic.nbranches):
				t = t.max(2/(dynamic.field(dynamic.grid[i+1]) - dynamic.field(dynamic.grid[i])))
			if n==1:
				return (dynamic.field(1), 2*dynamic.distorsion() + t, 2*lam, ) 
			else:
				return (dynamic.field(1), (2*dynamic.distorsion() + t) / (1-2*lam), 2*lam, ) 
		else:
			raise NotImplementedError("L-Y constants are implemented only for functions which are continuous on the torus. Define a method is_continuous_on_torus() if your map is so.")

	def projection_error(self):
		r"""
		Returns a bound on the projection error :math:`\|\Pi-I\|_{s\to w}`
		"""
		return partition_diameter(self.partition)

