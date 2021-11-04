"""
Functions to compute approximate eigenvectors of the Frobenius-Perron operator :math:`P` and assess their error.

Each entry ``v[i]`` of v is the integral of the (approximate) invariant measure the interval ``[partition[i], partition[i+1]]``. Hence ``sum(v)==1``. The (mean) value of the function on an interval is obtained dividing by the interval length.

Note: the right norm to use for ``v`` is the "applied mathematician's" one :math:`\|v\|_1 := \sum_i |v_i|`, not the "analyst's" one `\|v\|_{L^1} := \sum |I_i| |v_i|`.

Similarly, using hat functions, the "traditional" infinity norm on vectors is the correct one.
"""

from sage.all import vector, RealField, RR
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import numpy as np
import numpy.linalg as npl
import sys
from sparse import sage_sparse_to_scipy_sparse, sparse_matvec, interval_norm1_error, max_nonzeros_per_row, norm1, interval_norminf_error
from joblib import Parallel, delayed

__all__ = ["perron_vector", "rigorous_residual", "decay_time_hole"]

def perron_vector(P):
	"""
	Returns a (non-rigorous) approximation to the Perron eigenvector of ``P``, using ``scipy.sparse.linalg.eigs``.
	
	Args:
		P (interval Sage matrix):
	
	Returns:
		v (numpy array): Perron vector normalized to have norm-1 equal to 1
	"""

	P = P.apply_map(lambda x: RR(x.center()))
	P = sage_sparse_to_scipy_sparse(P)
	sys.stdout.write("Computing Perron vector...")
	sys.stdout.flush()
	v = eigs(P,1)[1][:,0]
	sys.stdout.write("done.\n")
	sys.stdout.flush()

	# sanitizing and normalization
	v = np.real(v)
	v = v / sum(v) # this is needed before v[v<0] = 0, so that almost all entries have positive sign
	v[v<0] = 0
	v = v / np.linalg.norm(v, ord=1) #probably faster than our norm1(v), since we don't need to control rounding
	return v

def rigorous_residual(P, v, norm='L_1'):
	"""
	Computes the norm-1 residual of ``v`` as the Perron vector of ``P``.
	
	Args:
		P (Sage interval matrix): 
		v: may be a lower-precision real vector, will be coherced to the same ring as P.
	"""
	
	assert all(v>=0)
	
	intervalfield = P.base_ring()
	
	#I tried to use change_ring() and apply_map() instead of these for loops, but they are too slow for large dimensions
	#Probably they insist on allocating too much
	
	#coerce v to the interval field
	vv = P.parent().column_space().dense_module()(0)
	for i in range(len(vv)):
		vv[i]=intervalfield(v[i])
	
	w = sparse_matvec(P, vv)
	
	# compute norm1 of the difference
	nm = intervalfield(0)
	if norm == 'L_1':
		for i in range(len(w)):
			nm += (w[i]-vv[i]).abs()
	else:
		if norm == 'L_inf':
			for i in range(len(w)):
				nm = nm.max(w[i]-vv[i])
		else:
			raise ValueError, 'Unknown norm'
	return nm
		
def decay_time_single_vector(P, v, target, steperror, norm):
	r"""
	Compute a ``N`` such that :math:`\|P^Nv\| \leq` ``target``.
	
	Args:
		P (numpy matrix):
		steperror (real): a value such that the error on the computed product :math:`Pv` is bounded by 
			``steperror*`` :math:`\|v\|`.
		my_norm (function): a function returning the (rigorous) norm to use.
	"""
	
	initialnorm = norm(v) #this might be known a-priori or computed twice, but whatever
	Rup = RealField(rnd='RNDU') #default precision for this error computation; more shouldn't be needed
	error = Rup(0.0)
	for i in range(1000):
		v = P * v
		currentnorm = norm(v)
		error += steperror * currentnorm
		if error + currentnorm < target:
			return i
		if error > 0.5*target:
			raise ValueError, "The numerical error is growing too much; raise the precision."
		if currentnorm > 5*initialnorm:
			raise ValueError, "The norm is growing too much; is this not contracting?"
		if i >= 999:
			raise ValueError, "Too many iterations; this looks strange."

def decay_time_helper_hole(P, k, target, steperror, norm):
	"""
	Helper function for parallelizing the decay time computation
	
	Compute decay time of the kth element of the basis returned by :func:`canonical_1k_basis`.
	
	k (int): basis element; it is the (zero-based) position of the entry -0.5, so it goes from 1 to n-1.
	"""
	n = P.shape[0]
	v = np.zeros(n)
	v[k] = 1
		
	return decay_time_single_vector(P, v, target, steperror, norm = norm)

def rignorm1(v):
	"""
	Compute an upper bound to the norm-1 of a numpy vector.
	"""
	return norm1(v, rnd='RNDU')

def rignorm_inf(v):
	"""
	Compute an upper bound to the infinity norm of a numpy vector
	"""
	return max(abs(v)) #max is not affected by rounding, so there is no need to take special tricks here

def rignorm_inf_C2(v): #not rigorous yet, due to the sums... 
	"""
	Compute an upper bound to the infinity norm of a function whose C2 spline coefficients are stored in v
	"""
	n = len(v)
	size = (n-1)/2
	a = RealInterval(max(abs(v[0:size])))
	b = RealInterval(max(abs(v[size:2*size])))/size
	a = a + b +abs(v[2*size])
	return a.upper()

def decay_time_hole(P, alpha = 0.5, n_jobs=1, norm='L_1'):
	r"""
	Compute a ``N`` such that :math:`\|P^N\| \leq \alpha`.
	
	Args:
		P (Sage interval matrix):
		n_jobs (int): number of jobs to use in the parallel computations.
		norm (str): norm to use: 'L_1' or 'L_inf'
	
	The function keeps track of both errors due to the interval nature of P
	and numerical errors in the computations.
	"""
	
	Rup = RealField(rnd='RNDU')
	Rdown = RealField(rnd='RNDD')

	if norm == 'L_1':
		Perror = interval_norm1_error(P)
		norm_function = rignorm1
		target = Rdown(alpha)* 0.5
	else:
		if norm == 'L_inf':
			Perror = interval_norminf_error(P)
			norm_function = rignorm_inf
			target = Rdown(alpha) / P.dimensions()[0] / 2 #we need an additional factor 0.5 here because the vectors that we use have inf-norm 0.5 rather than 1
		if norm == 'L_inf_C2':
			Perror = interval_norminf_C2_error(P)
			norm_function = rignorm_inf_C2
			target = Rdown(alpha) / P.dimensions()[0] / 2
		else:
			raise ValueError, 'Wrong norm specified'
	
	nnz = max_nonzeros_per_row(P)
	
	P = sage_sparse_to_scipy_sparse(P.apply_map(lambda x: RR(x.center())))
	n = P.shape[0]
	
	machineEpsilon = Rup(np.finfo(float).eps)
	gammak = machineEpsilon*nnz / (1-machineEpsilon*nnz)
	steperror = Rup(Perror) + Rup(gammak)
	
	sys.stdout.write('Computing decay time: iterating on each basis vector.\n')
	sys.stdout.flush()
	
	decayTimes = Parallel(n_jobs=n_jobs, verbose=1)(delayed(decay_time_helper_hole)(P,k,target = target, steperror = steperror, norm = norm_function) for k in range(n))
	
	return max(decayTimes)

