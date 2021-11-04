"""
Rewrite of the various assembler functions in a more generic fashion.
"""

from sage.all import matrix,RDF
from scipy.sparse import *
from huge_matrix import *


def assemble(dynamic, basis, epsilon, prec=53, scipy_matrix=False, huge_matrix_flag=False, output_rate=1024):
	"""
	Very generic assembler function
	"""
	
	print "Epsilon ", epsilon
	
	absolute_diameter=0
	
	n = len(basis)
	
	if scipy_matrix:
		P = lil_matrix((n,n))
	elif huge_matrix_flag:
		P = huge_matrix(n,n)
	else:
		P = matrix(dynamic.field, n, n, sparse=True)

	i = 0
	
	for dual_element in basis.dual_composed_with_dynamic(dynamic, epsilon):
		for j,x in basis.project_dual_element(dual_element):
			#if (x.absolute_diameter()>1024*epsilon):
			#	print i, j, x.absolute_diameter()
			if scipy_matrix:
				P[i,j] += RDF(x.center())
				absolute_diameter=max(absolute_diameter,x.absolute_diameter())
			elif huge_matrix_flag:
				P.sum_coefficients(i,j,RDF(x.center()))
				absolute_diameter=max(absolute_diameter,x.absolute_diameter())
			else:
				P[i,j] += x	
					
		i += 1
		if (output_rate is not 0) and (i%output_rate==0):
			print i
	
	if scipy_matrix:
		P=P.tocsr()
		print "absolute diameter", absolute_diameter
		
	if huge_matrix_flag:
		P.finished_assembling()
		print "absolute diameter", absolute_diameter
	
	return P
