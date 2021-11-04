"""
Rewrite of the various assembler functions in a more generic fashion.
"""

from sage.all import matrix
from warnings import warn

def assemble(dynamic, basis, epsilon, prec=53):
	"""
	Very generic assembler function; please note that now is contained in 
	the basis class
	"""
	warn('Deprecated: we moved assemble inside the basis class')
	
	n = len(basis)
	P = matrix(dynamic.field, n, n, sparse=True)
	
	for i, dual_element in basis.dual_composed_with_dynamic(dynamic, epsilon):
		for j, x in basis.project_dual_element(dual_element):
			P[i,j] += x
	
	return P
