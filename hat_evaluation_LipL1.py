from hat_evaluation import *

class HatFunctionsWithEvaluationsLipL1(HatFunctionsWithEvaluationsLinf):

	def ly_weak_to_weak_norm_equivalence(self):
		r"""
		This is the same bound as for the other bases, but this time it's non-trivial
		(since we need ||f||_1 \leq ||f||_inf), so we overload the function explicitly.
		"""
		return 1

	def lasota_yorke_constants(self, dynamic):
		warn('Deprecated: we are trying to move to dfly')

		raise NotImplementedError

		if not dynamic.is_continuous_on_torus():
			raise NotImplementedError

		BB = dynamic.weak_distorsion()

		alpha = (BB*2 + 1) / dynamic.expansivity()

		if not alpha < 1:
			raise ValueError("This map does not admit a Lasota-Yorke. Try with an iterate (see :func:`iterate_with_lasota_yorke`)")
		# from folder holes_filling in our notes
		return (alpha, BB*(BB+1))

	def semidiscrete_lasota_yorke_constants(self, dynamic):
		# the projection operators are *not* contractions, so we need some care...

		raise NotImplementedError



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

		if not dynamic.is_continuous_on_torus():
			raise NotImplementedError

		BB = dynamic.weak_distorsion()
		l = 1/dynamic.expansivity()
		B = BB / (1-l)


		if discrete is False:

			# from folder holes_filling in our notes

			if n is None:
			
				return (2*B+1, B*(B+1), l)

			elif n==1:
				alpha = (BB*2 + 1) * l

				if not alpha < 1:
					raise ValueError("This map does not admit a Lasota-Yorke. Try with an iterate (see :func:`iterate_with_lasota_yorke`)")

				return (dynamic.field(1), BB*(BB+1), alpha)			
			
			else:
				# if n is specified, we can truncate the sum 1+l+l^2+... earlier
				B = BB * (1-l**n) / (1-l)

				return (2*B+1, B*(B+1), l)

		else:
			raise NotImplementedError("Use the `small matrix' method instead")

	def projection_error_ly(self):
		r"""
		Returns a bound on the projection error :math:`\|\Pi-I\|_{s\to w'}` **with the LY norm w'**
		"""
		return partition_diameter(self.partition) / 4

	def small_matrix_bounds(self, dynamic, m):
		"""
		Simultaneous bounds to ||Q_h^i f||_s and ||Q_h^i f||_w, for i in range(m)

		From `matricine.pdf` in our notes

		Returns:
		  lips: list of bounds to ||f||_s, ||Q_h*f||_s, ... for a function with ||f||_w = 1
		  norms: list of bounds to ||f||_w, ||Q_h*f||_w, ...

		"""

		lips = vector(RealField(rnd='RNDU'), m)
		norms = vector(RealField(rnd='RNDU'), m)

		v = vector([self.strong_to_weak_norm_equivalence(), self.ly_weak_to_weak_norm_equivalence()])
		lips[0] = v[0]
		norms[0] = v[1]

		(A, B, l) = self.dfly(dynamic, n=1)

		if not A*l < 1:
			raise ValueError("This map does not admit a Lasota-Yorke. Try with an iterate (see :func:`iterate_with_lasota_yorke`)")

		M1 = matrix([[A*l, B], [0, 1]], ring=dynamic.field)
		M2 = matrix([[1, 0], [2*self.projection_error_ly(), 1]], ring=dynamic.field)

		for i in range(1, m):
			v = M2*(M1*v)
			lips[i] = v[0]
			norms[i] = v[0] + v[1] #TODO: not super-general

		return (lips, norms)

	def invariant_measure_strong_norm_bound(self, dynamic):

		# note that the "generic" bound in basis.py does not hold, since ||f||_inf is larger than 1.
		raise NotImplementedError

	def invariant_measure_strong_norm_bound(self, dynamic):
		"""
		A bound on the strong norm of the invariant measure of the dynamic.

		Typically this can be derived from the Lasota-Yorke constants.

		Returns:
			B' (real constant): such that :math:`B' \leq \|f\|_s`, :math:`f` being the invariant measure of the dynamic (normalized to have integral 1).
		"""

		A, B, l = self.dfly(dynamic)
		return B

	def normalization_norm(self):
		"""
		norm of the 'normalization operator' (subtract the integral)
		"""
		return RealField(rnd='RNDU')(2)

	def projection_error(self):
		return partition_diameter(self.partition) / 2

	def mixed_matrix_norm_approximation_error(self, dynamic):
		return  (self.bound_on_norms_of_powers(dynamic) + 1) * self.projection_error() * self.normalization_norm()
