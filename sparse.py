"""
Utility functions on sparse matrices.
Mostly created to work around Sage inefficiencies.
"""

from __future__ import division
from sage.rings.real_mpfr import RealField
from sage.rings.real_mpfi import is_RealIntervalFieldElement
from sage.all import VectorSpace, RR, vector
import numpy as np
from scipy.sparse import csr_matrix
from collections import Counter
from itertools import izip
from sage.all import RealNumber

__all__ = ['sparse_matvec', 'sparse_vecmat', 'interval_norm1_error', 'sage_sparse_to_scipy_sparse', 'max_nonzeros_per_row', 'norm1']

def sparse_matvec(P,v):
	"""
	Compute Sage sparse matrix * vector product.
	
	Apparently, M*v fills the matrix in Sage, so this should be quicker
	"""
	
	w = P.parent().column_space()(0)
	for (ij, val) in P.dict().iteritems():
		w[ij[0]] += val*v[ij[1]]
	return w
	
def sparse_vecmat(v,P):
	"""
	Compute Sage sparse vector * matrix product.
	
	Apparently, v*M fills the matrix in Sage, so this should be quicker
	"""
	
	w = P.parent().row_space()(0)
	for (ij, val) in P.dict().iteritems():
		w[ij[1]] += v[ij[0]] * val
	return w
	
	
def interval_norm1_error(P):
	"""
	Compute the "norm-1 width of the interval matrix P", i.e., an upper bound for :math:`\|P_1-P_2\|_1`, where the :math:`P_i` are matrices inside the interval :math:`P`.
	
	This is essentially the norm of P.upper()-P.lower().
	"""
	
	w = VectorSpace(RealField(rnd='RNDU'),P.dimensions()[1])(0)
	for (ij, val) in P.dict().iteritems():
		w[ij[1]] += val.absolute_diameter() #operations are rounded up because the LHS is so
	return max(w)

def interval_norminf_error(P):
	"""
	Compute the "norm-infinity width of the interval matrix P", i.e., an upper bound for :math:`\|P_1-P_2\|_inf`, where the :math:`P_i` are matrices inside the interval :math:`P`.
	
	This is essentially the norm of P.upper()-P.lower().
	"""
	
	w = VectorSpace(RealField(rnd='RNDU'),P.dimensions()[0])(0)
	for (ij, val) in P.dict().iteritems():
		w[ij[0]] += val.absolute_diameter() #operations are rounded up because the LHS is so
	return max(w)

def sage_sparse_to_scipy_sparse(P):
	"""
	Convert a Sage sparse matrix to a scipy CSR one without going through the full matrix.
	
	Method taken from http://www.sagenb.org/src/plot/matrix_plot.py.
	"""
	
	if is_RealIntervalFieldElement(P[0,0]):
		P = P.apply_map(lambda x: RR(x.center()))
	entries = list(P._dict().items())
	data = np.asarray([d for _,d in entries], dtype=float)
	positions = np.asarray([[row for (row,col),_ in entries],
		[col for (row,col),_ in entries]], dtype=int)
	return csr_matrix((data, positions), shape=(P.nrows(), P.ncols()))

def max_nonzeros_per_row(P):
	"""
	Max number of nonzeros in a row of the matrix P
	"""
	c = Counter(i for (i,j) in P.dict())
	return max(c.values())

def norm1(v, rnd='RNDN'):
	"""
	Compute the 1-norm of a vector, with configurable rounding mode.
	
	Returns a real with the correct rounding mode.
	"""
	
	nm = RealField(rnd=rnd)(0)
	for vi in v:
		nm += abs(vi) #operations are coherced to the ring of the LHS, so they are done with the correct rounding. Abs is exact in IEEE.
	return nm

def matrix_norm1(PP, rnd='RNDN'):
	"""
	Compute the 1-norm of a matrix, with configurable rounding mode
	
	Returns a real with the correct rounding mode.
	
	Args:
		PP (scipy sparse matrix):
	"""
	if not PP.format == 'csr':
		raise NotImplementedError
	column_norms = vector(RealField(rnd=rnd), PP.shape[1])
	for j, Pij in izip(PP.indices, PP.data):
		column_norms[j] += abs(Pij)
	return max(column_norms)
	
def matrix_norminf(PP, rnd='RNDN'):
	"""
	Compute the infinity norm of a matrix, with configurable rounding mode.

	Args:
		PP (scipy sparse matrix):
	"""

	if not PP.format == 'csr':
		raise NotImplementedError

	return max(norm1(PP.data[PP.indptr[i]:PP.indptr[i+1]], rnd=rnd) for i in xrange(PP.shape[0]))
	

