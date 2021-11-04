"""
Abstract base class for bases. All your bases are belong to us.
"""

from sage.all import RIF, vector
from sparse import sparse_matvec

class Basis:
	"""
	Defines the basis of a space used for approximating the invariant measure.
	
	This definition includes its dual basis, since they are intertwined together.
	"""
	
	def __len__(self):
		"""
		Number of items in the basis.
		"""
		raise NotImplementedError
	
	def dual_composed_with_dynamic(self, dynamic, epsilon):
		"""
		Generator function that returns L^*V_i for each V_i in the dual.
		
		L^*V_i is "represented" via a suitable object or data structure (depending on the basis chosen).
		For instance in the case of Ulam :math:`V_i(f) = \int_[x_{i},x_{i+1}] f`, and all we need to represent  :math:`L^*V_i` it is the two preimages :math:`T^{-1}(x_i),T^{-1}(x_{i+1})`.
		
		Returns:
			dual_element
		
		Abstract method.
		"""
		raise NotImplementedError

	def project_dual_element(self, dual_element, begin_i = 0, end_i = None):
		"""
		Computes the duality products of a dual element (as returned by :func:`dual_composed_with_dynamic`) with all the functions of the basis.
		
		Yields:
			pairs (j,x_j) such that :math:`V_i(L\Phi_j)=\sum_(k=j) x_k`, i.e., multiple pairs with the same j can be returned, in this case their x_j have to be summed.
		
		Abstract method.
		"""
		raise NotImplementedError

	def nonzero_on(self, I):
		r"""
		Generator yielding the set of indices i such that :math:`\phi_i(x) \neq 0` is nonzero for some `x\in I`
		
		Args:
			I (interval)
		"""
		raise NotImplementedError

	def evaluate(self, i, x):
		"""
		Evaluate the `i`th basis function on a point
		
		Args:
			x (real interval):
			i (int): index in `range(len(self))`
		"""
		raise NotImplementedError

	def evaluate_integral(self, i, ring=RIF):
		"""
		Integral of the `i`th basis function

		Args:
			ring (Sage ring): ring in which to coerce the partition
		"""

	def integral(self, v):
		"""
		Integral of a function in U_h

		Args:
			v (any type of vector):

		Returns:
			the integral, computed with the arithmetic of v.
		"""

		return sum(v[i]*self.evaluate_integral(i, ring=v[i].parent())  for i in range(len(self)))

	# norm-related functions

	def contracting_pairs(self):
		"""
		Yields a set of vectors on which to check contractivity.
		
		Yields:
			(u_i, s_i) (numpy vector, positive real): `len(self)-1` pairs such that:
		
		1. the u_i span the subspace of U_h with integral 0, and
		2. for each :math:`u=\sum b_i u_i`, we have :math:`\|u\| \geq \sum s_i |b_i|`
		"""
		raise NotImplementedError

	def norm_estimate(self, v):
		"""
		Rigorous estimate (from above) of ||v||
		
		Args:
			v (numpy vector):
			
		Returns:
			x (real with RNDU): such that :math:`\|v\| \leq v`
		"""
		raise NotImplementedError

	def matrix_norm_estimate(self, PP):
		"""
		Rigorous estimate (from above) of the matrix norm
		
		Args:
			PP (scipy sparse matrix):
			
		Returns:
			x (real with RNDU): such that :math:`\|PP\| \leq x`
		"""
		raise NotImplementedError

	def rigorous_norm(self, v):
		"""
		Rigorous norm of a vector.
				
		Args:
			v (Sage interval vector)
			
		Returns:
			its (weak) norm. 
		"""
		raise NotImplementedError

	def bound_on_norms_of_powers(self, dynamic, project_left = False, project_right = False):
		"""
		Uniform bound on :math:`\|L^i\|`.
		
		Gives a bound to the norms of :math:`\|(\Pi_1 L \Pi_2)^i\|` for each `i`, where
		:math:`\Pi_1` is the identity if `project_left==False`
		and the discretization projection if otherwise, and similarly for :math:`\Pi_2` and `project_right`.

		We do not use the terms "before" and "after" because they are ambiguous: would they refer to the order
		of function evaluations, or to the order in writing?

		Args:
			project_left, project_right (bools): tell whether to discretize.

		Returns:
			M (real with RNDU): such that :math:`\|(\Pi_1 L \Pi_2)^i\| \leq M` for each i
		
		"""
		raise NotImplementedError

	def matrix_norm_diameter(self, P):
		"""
		Diameter (in the matrix norm) of an interval matrix.
		
		Must be rigorous.
		
		Returns:
			M such that :math:`\|P_1-P_2\| \leq M` for all :math:`P_1,P_2 \in P`.
		"""
		raise NotImplementedError

	def residual_estimate(self,P, v):
		"""
		Computes the residual (in norm) of the computed Perron vector
		
		Args:
			P (interval matrix):
			v (numpy vector):
		
		Returns:
			res (real RNDU): an upper bound to :math:`\|Pv-v\|`
		"""
		v = P.parent().column_space()(vector(v))
		w = sparse_matvec(P, v)
		return self.rigorous_norm(v-w).upper()

	def invariant_measure_strong_norm_bound(self, dynamic):
		"""
		A bound on the strong norm of the invariant measure of the dynamic.

		Typically this can be derived from the Lasota-Yorke constants.

		Returns:
			B' (real constant): such that :math:`B' \leq \|f\|_s`, :math:`f` being the invariant measure of the dynamic.
		"""

		l, B = self.lasota_yorke_constants(dynamic)
		return B / (1-l)

	def iterate_with_lasota_yorke(self, dynamic):
		"""
		Return an iterate of `dynamic` (may be `dynamic` itself) which satisfies a Lasota-Yorke inequality.

		In some cases (most notably, in the infinity-norm), a dynamic `D` does not admit a LY inequality but an iterate `D^k` does.
		"""
		return NotImplementedError
