"""
Functions to refine partitions
"""

from __future__ import division
from sage.all import floor, log, sqrt
import numpy as np

from sage.all import load
load('binsearch.spyx')

def step_integral(v, partition, a, b):
	"""
	Integrate on ``[a,b]`` the step-function defined by ``partition`` and ``v``.
	
	``v`` contains the values of the integral of the function on each element, just like in the Ulam discretization.
	
	Non-rigorous.
	"""
	
	if len(v) + 1 != len(partition):
		raise ValueError, "Wrong partition size"
	
	if b<a:
		a, b = b, a
	if a<0 or b>1:
		raise ValueError, "The given interval is not in [0,1]"
	
	imin = binsearch(a, partition)
	assert imin>=0 and imin < len(partition) #we have changed the behavior of binsearch, checking that these edge cases were unused

	imax = binsearch(b, partition)
	assert imax>=0 and imax < len(partition) #we have changed the behavior of binsearch, checking that these edge cases were unused

	
	if imin == imax:
		return v[imin] * (b-a) / (partition[imin+1]-partition[imin])
	else:
		I1 = v[imin]*(partition[imin+1]-a)/(partition[imin+1]-partition[imin])
		I2 = v[imax]*(b-partition[imax])/(partition[imax+1]-partition[imax])
		return I1 + I2 + sum(v[i] for i in range(imin+1,imax))

def refine(v, partition, n, max_iterations=20):
	"""
	Given an approximation ``v`` of a measure on a partition ``partition``,
	construct a second partition that has between ``n/2`` and ``n`` intervals, each containing
	between ``1/n`` and ``2/n`` of the mass of ``v``, and each
	with length an inverse power of two.
	"""
	
	if len(v) + 1 != len(partition):
		raise ValueError, "Wrong partition size"
	
	mass = sum(v)
	
	new_partition=[0]
	size = 2.0 ** -floor(log(n,base=2))
	lower = 0
	while lower<1:
		while lower+size > 1:
			size /= 2
		integral = step_integral(v, partition, lower, lower+size)
		iterations = 0
		while integral > 2/n * mass:
			iterations += 1
			size /= 2
			integral = step_integral(v, partition, lower, lower+size)
			if iterations > max_iterations:
				raise ValueError, "Adaptive partitioning encountered very unequal partitions, or a bug. Raise max_iterations if it's the former."
		iterations = 0
		while integral <= 1/n * mass and lower+2*size <= 1:
			iterations += 1
			size *= 2
			integral = step_integral(v, partition, lower, lower+size)
			if iterations > max_iterations:
				raise ValueError, "Adaptive partitioning encountered very unequal partitions, or a bug. Raise max_iterations if it's the former."
				
		#at this point the integral should have the correct value
		new_partition.append(lower+size)
		lower = new_partition[-1]
	
	# The partition should already be 1-terminated at this point
	return np.asarray(new_partition)

def integral_distribution(v, partition, new_partition):
	"""
	Compute the integrals on the intervals of new_partition of the step function defined by ``v`` and ``partition``.
	"""
	
	return [step_integral(v, partition, new_partition[i], new_partition[i+1]) for i in range(len(new_partition)-1)]

def partition_sqrt(v, partition):
	"""
	Return the square root of a step function defined by ``v`` and ``partition``.
	
	Note that ``v`` contains integrals of the step function across the partition intervals,
	not function values, so it's more complicated than ``[sqrt(x) for x in v]``.
	"""
	
	w = np.zeros_like(v)
	for i in range(len(v)):
		w[i] = sqrt(v[i] * (partition[i+1]-partition[i]))
	return w
	

def sum_on_same_length(v, partition, newp):
	"""
	Given a step function defined by ``v`` and ``partition``, compute its total mass on each 
	family of intervals of the same length appearing in ``newp``.
	
	Returns:
		 (dictionary): partition length -> total mass on interval with that length.
	"""
	
	d = dict()
	for i in range(len(newp)-1):
		I = step_integral(v, partition, newp[i], newp[i+1])
		try:
			d[newp[i+1]-newp[i]] += I
		except KeyError:
			d[newp[i+1]-newp[i]] = I # create key if it doesn't exist
	return d

