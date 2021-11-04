"""
Ulam basis written in the generic assembler framework
"""

from __future__ import division
from basis import Basis
from partition import check_partition, partition_diameter, partition_minimum_diameter, is_refinement

from sage.all import load, RealNumber, RIF
load('binsearch.spyx')

import numpy as np
from sage.all import vector, RealField, RealIntervalField
from sparse import max_nonzeros_per_row, norm1, matrix_norm1, sparse_matvec

class Ulam(Basis):
	r"""
	Basis for the Ulam method
	
	Basis elements are the functions :math:`u_j = \frac{1}{|x_{j+1}-x_j|} 1_{[x_j,x_{j+1}]}`
	
	Dual basis elements are the integrals :math:`V_i = \int_{[x_i,x_{i+1}]}`.
	
	We have to normalize in this way because we want stochasticity in the resulting matrix, so we require \sum_i V_i to be the integral over the whole domain. Any other normalization would get into trouble when using unequal intervals.
	
	There is an important assumption that we are using at the moment: that all points have the same number of preimages and all branches of the function are monotonic.
	
	"""
	
	def __init__(self, partition):
		check_partition(partition)
		self.partition = partition
	
	def __len__(self):
		return len(self.partition) - 1
	
	def evaluate(self, i, x):
		# we need to cast to RealNumber manually here, because of a Sage bug when self.partition contains numpy floats (http://trac.sagemath.org/ticket/17758#ticket)
		field = x.parent()
		if x < RealNumber(self.partition[i]) or x > RealNumber(self.partition[i+1]):
			return field(0)
		elif x >= RealNumber(self.partition[i]) and x <= RealNumber(self.partition[i+1]):
			return field(1)
		else:
			return field(0,1)
		
	def nonzero_on(self, I):
		n = len(self)
		jmin = binsearch(I.lower(), self.partition)
		jmax = binsearch(I.upper(), self.partition)
		
		for j in range(jmin, jmax+1):
			yield j
			
	def dual_composed_with_dynamic(self, dynamic, epsilon):
		r"""
		For each `i`, yields the set of `dynamic.nbranches` pairs :math:`\{(f_k^{-1}(x_i),f_k^{-1}(x_{i+1})) \colon f_k \in \text{branch inverses}\}`.
		"""
		
		preimages_xi = list(dynamic.preimages(self.partition[0], epsilon))
		for i in range(len(self)):
			preimages_xi_plus_one = list(dynamic.preimages(self.partition[i+1], epsilon))
			yield zip(preimages_xi, preimages_xi_plus_one)
			preimages_xi = preimages_xi_plus_one
		
	def project_dual_element(self, dual_element):
		for a,b in dual_element:
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
		"""
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

	def semidiscrete_lasota_yorke_constants(self, dynamic):
		"""
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
		"""
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
		"""
		Return the constant required to estimate a strong norm with a weak norm (in the discretized space)

		Since we have a finite-dimensional discretized space, there is a constant
		such that :math:`\|f\|_s \leq C \|f\|_w` for each `f` in the discrete space.

		Returns:
			C (real with RNDU):
		"""
		
		return RealField(rnd='RNDU')(2) / partition_minimum_diameter(self.partition)
		
	def iterate_with_lasota_yorke(self, D):
		if not D.is_continuous_on_torus()
			raise NotImplementedError
		if D.expansivity() > 1:
			return D
		else:
			raise ValueError, "No iterate of this map admits a Lasota-Yorke inequality"
