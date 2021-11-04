"""
Finite element method based on hat functions + function evaluation as the dual basis.

For now, we throw in this file also the definition of the choices of an abstract finite element method (basis, norm)
"""

from partition import check_partition, partition_diameter, partition_minimum_diameter
from basis import Basis
from dynamic import IterateDynamic

import numpy as np
from sage.all import load, RIF, RealNumber, zero, RealField, vector
from sparse import max_nonzeros_per_row, matrix_norminf
load('binsearch.spyx')

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
		Evaluates a hat function on a point.
		"""

		field = x.parent()

		if i == 0:
			mi = field(self.partition[0])
			hi = field(self.partition[1])
			lo = field(self.partition[-2])
			mi2 = field(self.partition[-1])
			left_branch = (x-lo)/(mi2-lo)
			right_branch = (hi-x)/(hi-mi)
			return left_branch.max(right_branch).max(0)
		else:
			lo = field(self.partition[i-1])
			mi = field(self.partition[i])
			hi = field(self.partition[i+1])
			left_branch = (x-lo)/(mi-lo)
			right_branch = (hi-x)/(hi-mi)
			return left_branch.min(right_branch).max(0)

	def nonzero_on(self, I):
		n = len(self)
		jmin = binsearch(I.lower(), self.partition)
		jmax = binsearch(I.upper(), self.partition)
		
		for i in range(jmin, jmax+1):
			yield(i % n)

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
		For each `i`, yields (successively) the sequence :math:`\{(y,T'(y)) \colon y\in T^{-1}(x_i)\}`.
		"""
		for x in self.partition[:-1]:
			yield [ (y,dynamic.fprime(y)) for y in dynamic.preimages(x, epsilon) ]
		
	def project_dual_element(self, dual_element):
		for y, Tprimey in dual_element:
			for j in self.nonzero_on(y):
				yield (j, self.evaluate(j,y) / Tprimey.abs())

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
		nm = zero(v.base_ring())
		for x in v:
			nm = nm.max(x.abs())
		return nm

	def bound_on_norms_of_powers(self, dynamic, project_left = False, project_right = False):
		# After pondering [Galatolo, Nisoli] and their reference [3], I am convinced that this holds also for the projected ones. -federico
		return dynamic.distorsion() + 1

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
		"""
		Return Lasota-Yorke constants
		
		This is meant to replace `dynamic.lasota_yorke_constants()` with a more norm-agnostic packaging
		
		Returns:
			(lambda, B) (pair of real intervals): such that :math:`\|Lf\|_s \leq \lambda \|f\|_s + B\|f\|`
		"""

		if not dynamic.expansivity() > 1:
			raise ValueError, 'Your map is not expansive. A Lasota-Yorke inequality probably doesn''t exist.'

		if dynamic.is_continuous_on_torus():
			return (1/dynamic.expansivity(), dynamic.distorsion())
		else:
			raise NotImplementedError, "L-Y constants are implemented only for functions which are continuous on the torus. Define a method is_continuous_on_torus() if your map is so."

	def lasota_yorke_constants(self, dynamic):
		M = self.bound_on_norms_of_powers(dynamic)
		alpha = M / dynamic.expansivity()
		if not alpha < 1:
			raise ValueError, "This map does not admit a Lasota-Yorke. Try with an iterate (see :func:`iterate_with_lasota_yorke`)"
		# from [Galatolo, Nisoli, Lemma 17 with n=1]
		return (alpha, M*(1+dynamic.lipschitz_of_L1() / (1-alpha))) 

	def semidiscrete_lasota_yorke_constants(self, dynamic):
		# the projection operators are contractions, so everything works fine
		return self.lasota_yorke_constants(dynamic)

	def mixed_matrix_norm_approximation_error(self, dynamic):
		return partition_diameter(self.partition)* 2 * (self.bound_on_norms_of_powers(dynamic) + 1)

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
		l = 1 / D.expansivity()
		M = self.bound_on_norms_of_powers(D)
		# return self if it works
		if M*l < 1:
			return D
		if not l < 1:
			raise ValueError, "No iterate of this map admits a Lasota-Yorke"
		# we compute k in this way to avoid approximation errors. If k is very large, things are probably going to be slow later anyways
		k = 2
		while not M*(l **k) < 1:
			k = k + 1
		return IterateDynamic(D, k)
