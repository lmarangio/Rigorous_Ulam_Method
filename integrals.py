"""
Compute several functions related to integrals
"""

from __future__ import division
import unittest

def evaluate_half_hat(a0, a1, x):
	r"""
	Computes :math:`a(x)`, where :math:`a(x)` is the function which satisfies :math:`a(a0)=0`, :math:`a(a1)=1`, :math:`a(x)` is an affine map between ``a0`` and ``a1`` and a constant outside.
	
	a0, a1 and x are intervals.
	
	Note that ``a0`` may be smaller or larger than ``a1``, no order is imposed.
	"""
	
	affine_part = (x-a0)/(a1-a0)
	return affine_part.max(0).min(1)

def integrate_half_hat_product(a0, a1, b0, b1):
	r"""
	Computes :math:`\int a(x)b(x)`, where :math:`a(x)` and :math:`b(x)` are "half-hat functions" as in :func:`evaluate_truncated_linear`, and the integration interval is the intersection of their non-constness domains.
	
	``a0,a1,b0,b1`` are intervals.
	"""
	
	# compute the actual integration interval
	alo = a0.min(a1)
	ahi = a0.max(a1)
	blo = b0.min(b1)
	bhi = b0.max(b1)
	lo = alo.max(blo)
	hi = ahi.min(bhi)
	
	if hi<lo:
		return a0.parent()(0)
	
	# We use Simpson's rule, which is exact since we are integrating a quadratic function.
	evaluation_points = (lo, (lo+hi)/2, hi)
	vals = tuple(evaluate_half_hat(a0, a1, x)*evaluate_half_hat(b0, b1, x) for x in evaluation_points)
	return (hi-lo).abs()*(vals[0] + 4*vals[1] + vals[2])/6
	
def integrate_hat_product(a, b):
	"""
	Computes :math:`\int a(x)b(x)`, where :math:`a(x), b(x)` are hat functions given through their control points 
	
	Args:
		a, b (containers of 3 intervals): ``a`` contains :math:`(x_0,x_1,x_2)` so that :math:`a(x_0)=0, a(x_1)=1, a(x_2)=0`
			are the discontinuity points of the derivative of the hat function :math:`a(x)`.
	"""
	return integrate_half_hat_product(a[0],a[1],b[0],b[1]) + integrate_half_hat_product(a[2],a[1],b[0],b[1]) + \
		integrate_half_hat_product(a[0],a[1],b[2],b[1]) + integrate_half_hat_product(a[2],a[1],b[2],b[1])

class BasicTest(unittest.TestCase):
	"""
	Unit tests.
	"""
	def runTest(self):
		from sage.all import RIF
		self.assertTrue(1.0/3 in integrate_half_hat_product(RIF(0),RIF(1),RIF(0),RIF(1)))
		self.assertTrue(1.0/6 in integrate_half_hat_product(RIF(0),RIF(1),RIF(1),RIF(0)))
		self.assertTrue(integrate_half_hat_product(RIF(0),RIF(1),RIF(1),RIF(2)) == 0)
		hat1 = (RIF(0), RIF(1), RIF(2))
		hat2 = (RIF(3), RIF(2), RIF(1))
		self.assertTrue(2.0/3 in integrate_hat_product(hat1, hat1))
		self.assertTrue(1.0/6 in integrate_hat_product(hat1, hat2))
		self.assertTrue(1.0/6 in integrate_hat_product(hat2, hat1))
		
if __name__ == '__main__':
		unittest.main()

