"""
Abstract base class for bases. All your bases are belong to us.
"""

from sage.all import RIF, vector, matrix
from sparse import sparse_matvec
from warnings import warn

class Basis:
	"""
	Defines the basis of a space used for approximating the invariant measure.
	
	This definition includes its dual basis, since they are intertwined together.

	It also includes three norms: the "strong norm" (not really a norm), the "weak norm",
	and the "Lasota-Yorke weak norm" || ||_{w'} (which may be different, in the case of the Lip-L1 LY).

	"""
	
	def __len__(self):
		"""
		Number of items in the basis.
		"""
		raise NotImplementedError
	
	def dual_composed_with_dynamic(self, dynamic, epsilon):
		"""
		Generator function that yields "dual objects".
		
		L^*V_i is "represented" via a suitable object or data structure 
		(depending on the basis chosen).
		For instance in the case of Ulam :math:`V_i(f) = \int_{I_i} f`, 
		and :math:`L^*V_i = sum_{K: K is a preimage of I_i} \int_K`
		all we need to represent  a summand of :math:`L^*V_i` is the two preimages 
		:math:`T_k^{-1}(x_i),T_k^{-1}(x_{i+1})` on each branch.
		In this case, we yield (i, (T_k^{-1}(x_i),T_k^{-1}(x_{i+1}))).

		Yields:
			pairs (i, dual_fragment), where i is an integer, and dual_fragment is an abstract
			      object that can be used in project_dual_element. 
		
		Abstract method.
		"""
		raise NotImplementedError

	def project_dual_element(self, dual_element):
		"""
		Computes the duality products of a dual element (as returned by :func:`dual_composed_with_dynamic`) with all the functions of the basis.
		
		Yields:
			pairs (j,x_j) such that :math:`V_i(L\Phi_j)=\sum_(k=j) x_k`, i.e., multiple pairs with the same j can be returned, in this case their x_j have to be summed.
		
		Abstract method.
		"""
		raise NotImplementedError

	def assemble(self, dynamic, epsilon, prec=53):
		"""
		Very generic assembler function
		"""
	
		n = len(self)
		P = matrix(dynamic.field, n, n, sparse=True)
		
		for i, dual_element in self.dual_composed_with_dynamic(dynamic, epsilon):
			for j, x in self.project_dual_element(dual_element):
				P[i,j] += x
	
		return P

	def nonzero_on(self, I):
		r"""
		Indices i such that :math:`\phi_i(x) \neq 0` is nonzero for some :math:`x\in I`
		
		Args:
			I (interval)

		Returns / Yields:
			an iterable (generator or sequence)
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
		Rigorous estimate (from above) of ||v||_w
		
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
			x (real with RNDU): such that :math:`\|PP\|_w \leq x`
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
		r"""
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
		r"""
		Diameter (in the matrix norm) of an interval matrix.
		
		Must be rigorous.
		
		Returns:
			M such that :math:`\|P_1-P_2\|_w \leq M` for all :math:`P_1,P_2 \in P`.
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
		w = sparse_matvec(P, v) - v
		w = w.apply_map(lambda x: x.magnitude())  # assumes the norm is monotonic
		return self.norm_estimate(w)

	def invariant_measure_strong_norm_bound(self, dynamic):
		"""
		A bound on the strong norm of the invariant measure of the dynamic.

		Typically this can be derived from the Lasota-Yorke constants.

		Returns:
			B' (real constant): such that :math:`B' \leq \|f\|_s`, :math:`f` being the invariant measure of the dynamic (normalized to have integral 1).
		"""

		A, B, l = self.dfly(dynamic)
		return B

	def lasota_yorke_constants(self, dynamic):
		"""
		Return Lasota-Yorke constants
		
		This is meant to replace `dynamic.lasota_yorke_constants()` with a more norm-agnostic packaging
		
		Returns:
			(lambda, B) (pair of real intervals): such that :math:`\|Lf\|_s \leq \lambda \|f\|_s + B\|f\|`
		"""
		warn('Deprecated: we are trying to move to dfly')
		raise NotImplementedError

	def iterate_with_lasota_yorke(self, dynamic):
		"""
		Return an iterate of `dynamic` (may be `dynamic` itself) which satisfies a Lasota-Yorke inequality.

		In some cases (most notably, in the infinity-norm), a dynamic `D` does not admit a LY inequality but an iterate `D^k` does.
		"""
		warn('Deprecated: we are trying to move to dfly')
		raise NotImplementedError

	def dfly(self, dynamic, discrete=False, n=None):
		"""
		Return constants of the dfly inequality :math:`\|L^n f\|_s \leq A \lambda^n \|f\|_s + B\|f\|_w` 

		This function should make `lasota_yorke_constants` and `iterate_with_lasota_yorke` obsolete.
		
		Input:
			Dynamic:
			discrete: if True, returns constants such that the DFLY holds for the projected operator 
			          and **for all functions in the discrete space only** (so we can make a DFLY for Pi*L instead of Pi*L*Pi)
			n: if it is not None, returns constants that hold only up to a certain value of n

		Returns:
			(A, B, lambda): tuple of intervals
		"""
		raise NotImplementedError

	def strong_to_weak_norm_equivalence(self):
		r"""
		Return the constant required to estimate a strong norm with a weak norm (in the discretized space)

		Since we have a finite-dimensional discretized space, there is a constant
		such that :math:`\|f\|_s \leq C \|f\|_w'` for each `f` in the discrete space.

		Returns:
			C (real with RNDU):
		"""
		raise NotImplementedError

	def ly_weak_to_weak_norm_equivalence(self):
		r"""
		Returns a constant such that ||f||_{w'} \leq ||f||_{w}. It's 1 in most cases.
		"""
		return 1

	def strong_power_norms(self, dynamic, m):
		"""
		returns a list of bounds ||L_h^i f||_s <= M_i*||f||_w, for i in range(m), which hold for all zero-integral functions f
		in the discretized space.
		"""

		L = list()

		K = self.strong_to_weak_norm_equivalence()
		K2 = self.ly_weak_to_weak_norm_equivalence()
		L.append(K)

		for i in range(1, m):
			(A, B, l) = self.dfly(dynamic, True, i)
			L.append(A*(l**i)*K + B*K2)

		return L
