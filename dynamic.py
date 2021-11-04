"""
Defines basic sample classes representing the dynamics.

In many of these examples, it is useful to preallocate the constants that we need as intervals, for performance reasons.
"""

from __future__ import division
from sage.rings.real_mpfi import RealIntervalField, is_RealIntervalFieldElement
from interval import interval_newton
from sage.all import floor

from warnings import warn
from sage.all import load


def warn_if_not_interval(x):
	"""
	Helper function that raises a warning if its argument is not the member of a Sage interval field.
	"""
	if not is_RealIntervalFieldElement(x):
		warn("The dynamic map was called with an argument which is not an interval, the results are not provably accurate.")

def warn_if_not_in_01(x):
	"""
	Helper function that raises a warning if its argument is not in 01
	"""
	if not 0 <= x <= 1:
		warn("The dynamic map was called with an argument which is not in [0,1].")

def mod1(I):
	"""
	Compute the fractional part of a value.
	
	``I`` could be real or an element of a Sage interval field.
	Returns an error if the interval is not strictly included in some :math:`[k,k+1]` for :math:`k\in\mathbb{Z}` (e.g., if you call ``mod1(RIF(0.9,1.1))``
	"""
	
	# so that function calls work also with non-interval reals
	if not is_RealIntervalFieldElement(I):
		warn_if_not_interval(I)
		return I - floor(I)
	
	k = I.lower().floor()
	if k+1 < I.upper().ceil():
		raise ValueError, "The given interval is not included in an interval of the form [k, k+1]"
	return I-k

class Dynamic:
	r"""
	Generic class representing dynamics.
	
	The following methods should be implemented by inherited classes:
	``f``, ``fprime``, ``expansivity()`` (returns :math:`\inf |f'|`) and ``distorsion()`` (returns :math:`\sup \frac{|f''|}{(f')^2}`)
	"""

	def lasota_yorke_constants(self, type):
		r"""
		Compute Lasota-Yorke constants.
		
		Args:
			type: 'L1' or 'Linf' for now
		
		Returns: 
			(lambda, B) (pair of interval field members): such that a Lasota-Yorke
                        inequality :math:`\|Lf\|_s \leq \lambda \|f\|_s + B\|f\|_w`.
                        The two norms depend on the parameter ``type``.
		"""
		
		if not self.is_continuous_on_torus():
			raise ValueError, "L-Y constants are implemented automatically only for functions which are continuous on the torus. Define a method is_continuous_on_torus() if your map is so."
		
		if not self.expansivity() > 1:
			raise ValueError, 'Your map is not expansive. A Lasota-Yorke inequality probably doesn''t exist.'
		
		if type == 'L1':
			if not self.is_continuous_on_torus():
				raise NotImplementedError, "We haven't done this yet"
			return (1/self.expansivity(), self.distorsion())
		elif type == 'Linf' or type == 'C0':
			M = self.bound_on_norms_of_powers('Linf')
			alpha = M / self.expansivity() 
			if not alpha < 1:
				raise ValueError, "This map does not admit a Lasota-Yorke. Try with an iterate (see :func:`iterate_with_lasota_yorke`)"
			# from [Galatolo, Nisoli, Lemma 17 with n=1]
			return (alpha, M*(1+self.lipschitz_of_L1() / (1-alpha))) 
			
			raise NotImplementedError
		else:
			raise ValueError, 'Unknwon Lasota-Yorke type'
			
	def iterate_with_lasota_yorke(self, type):
		r"""
		return an iterate of this dynamic which admits a Lasota-Yorke of the given type.
		"""
		
		if type == 'L1' and self.is_continuous_on_torus():
			return self
		elif type == 'L1' and not self.is_continuous_on_torus():
			raise NotImplementedError, "The LY coefficients probably need some adjustment here, double-check"
		elif type == 'Linf' or type == 'C0':
			lambd = 1 / self.expansivity()
			M = self.bound_on_norms_of_powers('Linf')
			# return self if it works
			if M*lambd < 1:
				return self
			if not lambd < 1:
				raise ValueError, "No iterate of this map admits a Lasota-Yorke"
			# we compute k in this way to avoid approximation errors. If k is very large, things are probably going to be slow later anyways
			k = 2
			while not M*(lambd **k) < 1:
				k = k + 1
			return IterateDynamic(self, k)
		else:
			raise ValueError, "Unknown Lasota-Yorke type"
	
	def bound_on_norms_of_powers(self, type):
		r"""
		uniform bound on the norm of L^n
		
		Return M such that :math:`\|L^n\|\leq M` for each :math:`n\in\mathbb{N}`.
		
		Args:
			type (string): 'L1' or 'Linf' (synonym: 'C0')
			
		"""
		
		if type == 'L1':
			return 1
		elif type == 'Linf' or type == 'C0':
			lam=1/self.expansivity()
			return 1+(self.distorsion())/(1-lam)
		else:
			raise ValueError, 'Unknown norm type'

	def preimages(self, x, epsilon):
		"""
		Generator function returning all preimages of a given value.
		
		Args:
			x (interval):
			epsilon (real): tries to obtain a preimage interval smaller than epsilon
			
		Yields:
			intervals (typically of width `epsilon` or smaller) containing all preimages of `x`.
			
		The long-term plan is eliminating or changing the concept of "branches" and using this function
		to replace all calls to :func:`preimage()`
		"""
		
		for b in range(self.nbranches):
			yield self.preimage(x, b, epsilon)

	def perron_operator(self, f, x, epsilon=1e-10):
		"""
		Evaluates the Perron operator for a density function f
		"""
		
		return sum(f(y)/self.fprime(y).abs()  for y in self.preimages(x, epsilon))


class Mod1Dynamic(Dynamic):
	r"""
	Generic class representing the dynamic of a full-branch increasing function :math:`f(x) \mod 1`.
	
	We assume that :math:`f(0)=0`, :math:`f(1)=k`, where ``k==self.nbranches``.
	
	The following methods should be implemented by inherited classes:
	``f_unquotiented``, ``f``, ``fprime``, ``expansivity()`` (returns :math:`\inf |f'|`) and ``distorsion()`` (returns :math:`\sup \frac{|f''|}{(f')^2}`)
	"""
	def __init__(self, nbranches, prec):
		self.field = RealIntervalField(prec=prec)
		self.nbranches = nbranches
		self._domain = self.field('0','1')
	
	def f(self, x):
		"""
		Applies the function
		"""
		warn_if_not_in_01(x)
		return mod1(self.f_unquotiented(x))
	
	def preimage(self, x, branch, epsilon):
		"""
		Compute the preimage of a point (in interval arithmetic) on a given branch.
		
		``branch`` should be in ``range(self.nbranches)``.
		
		The argument x gets coerced to self.field.
		"""

		warn_if_not_in_01(x)
		unquotiented_x = self.field(x) + branch 
		one = self.field(1)
		
		f = lambda(x): self.f_unquotiented(x)
		fprime = lambda(x): self.fprime(x)
		return interval_newton(f, fprime, self._domain, unquotiented_x, epsilon)
		
	def f_unquotiented(self, x):
		"""
		Not implemented, derived classes must specialize this.
		"""
		raise NotImplementedError
	
	def fprime(self, x):
		"""
		Not implemented, derived classes must specialize this.
		"""
		raise NotImplementedError
	
	def lipschitz_of_L1(self):
		"""
		Bound on the Lipschitz constant of L applied to the identically-one function.
		"""
		return self.nbranches * self.distorsion()

class QuotientedLinearDynamic(Mod1Dynamic):
	r"""
	Represent the dynamic of the map :math:`x \mapsto kx \mod 1`.
	
	:math:`k` is not constrained to be an integer; if it is not, then the last branch "ends" before 1.
	"""
	def __init__(self, k, prec=53):
		# hack: we need to provide an nbranches to the constructor of Mod1Dynamic, but
		# we need self.field to compute it; so we set it to -1 and adjust it later.
		Mod1Dynamic.__init__(self, -1, prec)
		self._slope = self.field(k)
		self._is_continuous = self._slope.unique_floor() == self._slope.unique_ceil()
		self.nbranches = self._slope.unique_ceil()
	
	def f_unquotiented(self, x):
		x = self.field(x)
		y = self._slope * x
		if (not self._is_continuous) and self.field(1).overlaps(x):
			y = y.union(self.nbranches)
		return y
		
	def fprime(self, x):
		if (not self._is_continuous):
			if x == self.field(1):
				return self.field('+inf')
			elif self.field(1).overlaps(x):
				return self._slope.union(self.field('+inf'))
			else:
				return self._slope
		else:
			return self._slope
		
	def fsecond(self, x):
		return self._intervalfield(0)

	def expansivity(self):
		return self._slope
		
	def distorsion(self):
		return self.field(0)
	
	def is_continuous_on_torus(self):
		return self._is_continuous

class MannevillePomeauDynamic(Mod1Dynamic):
	r"""
	dynamic of the Manneville-Pomeau map :math:`x\mapsto x+x^{1+\alpha}`.
	
	"""
	def __init__(self, alpha=1.0/16, prec=128):
		"""
		Args:
			alpha (real or interval): 
			prec (int): number of precision digits
		"""
		Mod1Dynamic.__init__(self, 2, prec)
		self._alpha = self.field(alpha)
		self._one = self.field('1')
		self._alphaplusone = self._alpha + self._one
			
	def f_unquotiented(self, x):
		"""
		"""
		warn_if_not_interval(x)
		warn_if_not_in_01(x)
		return x+x**self._alphaplusone
		
	def fprime(self, x):
		"""
		"""
		warn_if_not_interval(x)
		warn_if_not_in_01(x)
		return self._one + self._alphaplusone * (x ** self._alpha)

	def expansivity(self):
		return self._one

	def is_continuous_on_torus(self):
		return True

class PerturbedFourxDynamic(Mod1Dynamic):
	"""
	Dynamic :math:`4x+c\sin(j\pi x) \mod 1`, for tunable parameters ``c`` and ``j``.
	"""
	def __init__(self, prec=53, c='0.01', j=2):
		"""
		Args:
			
		"""
		Mod1Dynamic.__init__(self, 4, prec)
		self._four = self.field('4')
		self._c = self.field(c)
		self._jpi = self.field.pi() * j
		self._j=j
		infTprime = self._four - self._c*self._jpi
		if not infTprime > 1:
			warn("The dynamic is not expanding with this choice of the parameters")
	
	def f_unquotiented(self, x):
		"""
		"""
		warn_if_not_interval(x)
		warn_if_not_in_01(x)
		y=self._four * x + self._c * (self._jpi*x).sin()
		#if (self._j%4==0): #since points of the  type i/4 are sent into integers, we can use this information to get better bounds on the computation  
		#	if ((x.upper()*self._j).is_integer()):
		#		a=y.unique_integer()
		#		y=y.intersection(self._intervalfield(a-1,a))
		#	if ((x.lower()*self._j).is_integer()):
		#		a=y.unique_integer()
		#		y=y.intersection(self._intervalfield(a-1,a))
		return y 
		
	def fprime(self, x):
		warn_if_not_interval(x)
		warn_if_not_in_01(x)
		return self._four + self._c * self._jpi * (self._jpi * x).cos()

	def expansivity(self):
		return self._four - self._c*self._jpi

	def distorsion(self):
		return self._c * (self._jpi**2) / self.expansivity()**2
	
	def third_distorsion(self):
		return self._c * self._jpi * self._jpi* self._jpi / self.expansivity()**3

	def is_continuous_on_torus(self):
		return True
	
	def fsecond(self, x):
		warn_if_not_interval(x)
		return -self._c * (self._jpi**2) * (self._jpi * x).sin()

	def fthird(self, x):
		warn_if_not_interval(x)
		return -self._c * (self._jpi**3) * (self._jpi * x).cos()

class PerturbedNxDynamic(Mod1Dynamic):
	"""
	Dynamic :math:`N(x+d1)+c\sin(2j\pi x + d2) \mod 1`, for tunable parameters ``c`` and ``j``.
	"""
	def __init__(self, prec=53, N='4', c='0.01', j=2, d1='0', d2='0'):
		"""
		Args:
			
		"""
		Mod1Dynamic.__init__(self, int(N), prec)
		self._d1 = self.field(d1)
		self._d2 = self.field(d2)
		self._N = self.field(N)
		self._c = self.field(c)
		self._j2pi = self.field.pi() * 2 * j
		infTprime = self._N - self._c*self._j2pi
		if not infTprime > 1:
			warn("The dynamic is not expanding with this choice of the parameters")

	def f_unquotiented(self, x):
		"""
		"""
		warn_if_not_interval(x)
		warn_if_not_in_01(x)
		return self._N * (x + self._d1) + self._c * (self._j2pi * x + self._d2).sin()

	def fprime(self, x):
		warn_if_not_interval(x)
		warn_if_not_in_01(x)
		return self._N + self._c * self._j2pi * (self._j2pi * x + self._d2).cos()

	def expansivity(self):
		return self._N - self._c*self._j2pi

	def distorsion(self):
		return self._c * (self._j2pi**2) / self.expansivity()**2

	def is_continuous_on_torus(self):
		return True
	
	def fsecond(self, x):
		warn_if_not_interval(x)
		return -self._c * (self._j2pi**2) * (self._j2pi * x + self._d2).sin()

class MongevillePoleauDynamic(Mod1Dynamic):
	r"""
	A home-made variant of Manneville-Pomeau, :math:`x\mapsto (1+\beta)x+(3-\beta)x^(1+\alpha) \mod 1`.
	"""

	def __init__(self, prec=53, alpha=1, beta=1/8):
		"""
		"""
		if beta<0 or beta>1 or alpha<1:
			raise ValueError, "Wrong parameters"
		Mod1Dynamic.__init__(self, 4, prec)
		self._alpha = self.field(alpha)
		beta = self.field(beta)
		one = self.field('1')
		self._oneplusbeta = beta + one
		self._oneplusalpha = self._alpha + one
		self._threeminusbeta = self.field('3') - beta
		infTprime = self._oneplusbeta
		if not infTprime > 1:
			raise ValueError, "This dynamic is not expanding"

	def expansivity(self):
		return self._oneplusbeta
	
	def distorsion(self):
		return self._threeminusbeta*self._oneplusalpha / self.expansivity()**2

	def f_unquotiented(self, x):
		"""
		"""
		warn_if_not_interval(x)
		warn_if_not_in_01(x)
		return self._oneplusbeta*x+self._threeminusbeta*(x**self._oneplusalpha)
		
	def fprime(self, x):
		"""
		"""
		warn_if_not_interval(x)
		warn_if_not_in_01(x)
		return self._oneplusbeta + self._threeminusbeta* self._oneplusalpha * (x**self._alpha)

	def is_continuous_on_torus(self):
		return True

class IterateDynamic(Dynamic):
	"""
	Class representing the iterate of another existing dynamic.
	"""
	def __init__(self, D, k):
		"""
		Constructor.
		"""
		self._k = k
		self._D = D
		self.nbranches = D.nbranches ** k
		self.field = D.field

	def expansivity(self):
		return self._D.expansivity() ** self._k
	
	def distorsion(self):
		
		if not self._D.is_continuous_on_torus():
			raise ValueError, 'TODO: double-check that this bound holds also for non-Markov functions'
		# Isaia proved that B_iterate \leq k*B; it should not be difficult by induction on k
		return self._D.distorsion() * self._k
		
	def bound_on_norms_of_powers(self, type):
		# the old bound is still valid, and it will typically be smaller
		return self._D.bound_on_norms_of_powers(type)
	
	def f(self, x):
		for i in range(self._k):
			x = self._D.f(x)
		return x
		
	def fprime(self, x):
		dfx = 1
		fx = x
		k = 0
		while True:
			dfx = self._D.fprime(fx) * dfx
			k = k+1
			if k < self._k:
				fx = self._D.f(fx)
			else:
				break
		return dfx
	
	def fsecond(self,x):
		ddfx = 0
		dfx = 1
		fx = x
		k = 0
		while True:
			dfx = self._D.fprime(fx) * dfx
			ddfx = self._D.fsecond(fx) * (dfx**2)+self._D.fprime(fx) * ddfx
			k = k+1
			if k < self._k:
				fx = self._D.f(fx)
			else:
				break
		return dfx
	
	def preimage(self, x, branch, epsilon):
		"""
		Note: branches are not listed in increasing order. It would be possible to do it that way, but the performance would be slightly worse.
		
		Note: it is not easy to compute the preimage up to a prescribed precision epsilon. So we do not attempt to do it here; rather we take epsilon as an indication. In other words: this will **not** always yield an interval of size smaller than epsilon; don't count on it.
		"""
		
		for i in range(self._k):
			current_branch = branch % self._D.nbranches
			x = self._D.preimage(x, current_branch, epsilon/self._k)
			branch = branch // self._D.nbranches
		return x
		
	def is_continuous_on_torus(self):
		return self._D.is_continuous_on_torus()
	
	def lipschitz_of_L1(self):
		"""
		Bound on the Lipschitz constant of L applied to the identically-one function.
		"""
		return self.nbranches * self.distorsion()


# Please note, that this Iterated class works under the hypothesis that T(x+k)=T(x)
# for every integer $k$. This, in particular implies that
# T'(mod1(T_unquotiented(x)))=T'(T_unquotiented(x)) and
# that mod1(T_unquotiented^n(x))=T^n(x)

class IterateMod1Dynamic(Mod1Dynamic):
	"""
	Class representing the iterate of another existing dynamic.
	"""
	def __init__(self, D, k):
		"""
		Constructor.
		"""
		self._k = k
		self._D = D
		self.nbranches = D.nbranches ** k

	def expansivity(self):
		return self._D.expansivity() ** self._k
	
	def distorsion(self):
		
		if not self._D.is_continuous_on_torus():
			raise ValueError, 'TODO: double-check that this bound holds also for non-Markov functions'
		# Isaia proved that B_iterate \leq k*B; it should not be difficult by induction on k
		return self._D.distorsion() * self._k
		
	def bound_on_norms_of_powers(self, type):
		# the old bound is still valid, and it will typically be smaller
		return self._D.bound_on_norms_of_powers(type)
	
	def f(self, x):
		for i in range(self._k):
			x = self._D.f_unquotiented(x)
		return mod1(x)
		
	def fprime(self, x):
		dfx = 1
		fx = x
		k = 0
		while True:
			dfx = self._D.fprime(fx) * dfx
			k = k+1
			if k < self._k:
				fx = self._D.f_unquotiented(fx)
			else:
				break
		return dfx
	
	def fsecond(self,x):
		ddfx = 0
		dfx = 1
		fx = x
		k = 0
		while True:
			dfx = self._D.fprime(fx) * dfx
			ddfx = self._D.fsecond(fx) * (dfx**2)+self._D.fprime(fx) * ddfx
			k = k+1
			if k < self._k:
				fx = self._D.f_unquotiented(fx)
			else:
				break
		return ddfx
	
	def preimage(self, x, branch, epsilon):
		"""
		Note: branches are not listed in increasing order. It would be possible to do it that way, but the performance would be slightly worse.
		
		Note: it is not easy to compute the preimage up to a prescribed precision epsilon. So we do not attempt to do it here; rather we take epsilon as an indication. In other words: this will **not** always yield an interval of size smaller than epsilon; don't count on it.
		"""
		
		for i in range(self._k):
			current_branch = branch % self._D.nbranches
			x = self._D.preimage(x, current_branch, epsilon/self._k)
			branch = branch // self._D.nbranches
		return x
		
	def is_continuous_on_torus(self):
		return self._D.is_continuous_on_torus()
	
	def lipschitz_of_L1(self):
		"""
		Bound on the Lipschitz constant of L applied to the identically-one function.
		"""
		return self.nbranches * self.distorsion()


class LanfordDynamic(Mod1Dynamic):
	r"""
	Lanford map :math:`x\mapsto 2x+\frac12x(1-x) \mod 1`.
	
	One of the examples in Galatolo-Nisoli.
	"""
	def __init__(self, prec = 53):
		Mod1Dynamic.__init__(self, 2, prec)
		self._five_over_two = self.field('2.5')
		self._one_over_two = self.field('0.5')

	def expansivity(self):
		return self.field('1.5')
	def distorsion(self):
		return self.field('4')/self.field('9')

	def f_unquotiented(self, x):
		"""
		"""
		warn_if_not_interval(x)
		warn_if_not_in_01(x)
		# Horner method
		return x*(self._five_over_two - self._one_over_two*x)
		
	def fprime(self, x):
		warn_if_not_interval(x)
		warn_if_not_in_01(x)
		return self._five_over_two - x

	def is_continuous_on_torus(self):
		return True
	
class ShiftedDynamic(Dynamic):
	"""
	Defines a dynamic obtained by shifting states by a constant :math:`c`.

	States are shifted both on start and on arrival, so the invariant measure should be a shifted version of the original one.
	
	This dynamic is useful for testing, to check if an unexpected dip/bump in the graph is real or if it is a numerical artifact/error.
	"""
	def __init__(self, D, c, prec=53):
		self._D = D
		intervalfield = RealIntervalField(prec=prec)
		self._c = intervalfield(c)
		self.nbranches = D.nbranches

	def f(self, y):
		return mod1(self._D.f(mod1(y-self._c)) + self._c)

	def fprime(self, y):
		return self._D.fprime(mod1(y-self._c))

	def preimage(self, y, branch, epsilon):
		return mod1(self._D.preimage(mod1(y-self._c), branch, epsilon)+self._c)

	def expansivity(self):
		return self._D.expansivity()
	def distorsion(self):
		return self._D.distorsion()

	def is_continuous_on_torus(self):
		return self._D.is_continuous_on_torus()

