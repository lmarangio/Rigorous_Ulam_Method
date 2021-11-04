"""
Functions to work with intervals (only Newton remained for now, there used to be other stuff but it's obsolete)
"""

class NoZeroException(Exception):
	pass

def interval_newton(f, fprime, I, alpha, epsilon):
	"""
	Finds a smaller interval inside I where f(x)-alpha has a zero, i.e., when f(x)=alpha. 
	f is supposed to be monotonic. f and fprime must take and return intervals.

	Stops when (if) the interval has width smaller than epsilon.
	Raises ValueError (intersection of non-overlapping intervals) if the function can be proved
	to have no zero in the interval.
	Returns an interval containing I.lower() or I.upper() if the function *may* have no zero
	in the interval.
	"""
	intervalfield = I.parent()
	
	alpha = intervalfield(alpha)

	for iterations in range(100):
		m = intervalfield(I.center())
		y = m - (f(m)-alpha)/fprime(I)
		try:
			J = I.intersection(y)
		except ValueError:
			raise NoZeroException
		if J.absolute_diameter() <= epsilon:
			return J
		if J == I:
			return J
		I = J
