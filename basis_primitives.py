from sage.rings.real_mpfr import RealField
from sage.all import RealNumber, NaN, floor
from sage.rings.real_mpfi import RealIntervalField, is_RealIntervalFieldElement
from sage.matrix.matrix_space import MatrixSpace
from sage.modules.free_module import VectorSpace
from joblib import Parallel, delayed
from sparse import sage_sparse_to_scipy_sparse

import numpy as np


import sys

_all__=["phi","phi_prime","tau","tau_prime","psi","psi_prime","from_derivative_at_nodes_to_compact_liverani_basis"
		,"compute_integral_liverani_basis","test_matrix","compute_M","prod_mat_vec"]

"""
	Implements the elements of the partition of unity
"""


"""
In this file, differently from before, we will denote by phi the elements
of the partition of unity, by tau their primitive and
by psi the difference between two neighobouring derivative
"""

"""A function that implements the Horner Scheme for evaluating a
    polynomial of coefficients *polynomial in x.
	From wikipedia...
"""

def horner(x, *polynomial):
    result = x.parent()(0)
    for coefficient in polynomial:
        result = result * x + coefficient
    return result

"""
The branches are
phi-(x)=6*x^5+15*x^4+10*x^3+1
phi+(x)=-6*x^5+15*x^4-10*x^3+1
"""
def phi_base(x):
	Interval_Field=x.parent()
	tilde_x=x
	#if x is not in [-1,1]
	try:
		tilde_x=tilde_x.intersection(Interval_Field(-1,1))
	except ValueError:
		return Interval_Field(0)
	try:
		tilde_x_neg=tilde_x.intersection(Interval_Field(-1,0))
		val_neg = horner(x,6,15,10,0,0,1)
	except ValueError:
		val_neg=None
	try:
		tilde_x_pos=tilde_x.intersection(Interval_Field(0,1))
		val_pos = horner(x,-6,15,-10,0,0,1)
	except ValueError:
		val_pos=None
	if val_pos is None:
		return val_neg
	if val_neg is None:
		return val_pos
	return val_neg.union(val_pos)

def phi(x,i,partsize):
	x_tilde=x*partsize-i
	return phi_base(x_tilde)

def phi_prime_base(x):
	Interval_Field=x.parent()
	tilde_x=x
	#if x is not in [-1,1]
	try:
		tilde_x=tilde_x.intersection(Interval_Field(-1,1))
	except ValueError:
		return Interval_Field(0)
	try:
		tilde_x_neg=tilde_x.intersection(Interval_Field(-1,0))
		val_neg = (horner(x,30,60,30,0,0))
	except ValueError:
		val_neg=None
	try:
		tilde_x_pos=tilde_x.intersection(Interval_Field(0,1))
		val_pos = (horner(x,-30,60,-30,0,0))
	except ValueError:
		val_pos=None
	if val_pos is None:
		return val_neg
	if val_neg is None:
		return val_pos
	return val_neg.union(val_pos)

def phi_prime(x,i,partsize):
	x_tilde=x*partsize-i
	return phi_prime_base(x_tilde)*partsize

def eta(x,partsize):
	y=(x.parent())(0)
	for i in range(partsize*2):
		 y+=(i%2)*phi(x,i,2*partsize)
	return 2*y



""" 
Implements the primitives of phi
x^6+3*x^5+5/2*x^4+x+0.5
-x^6+3*x^5-5/2*x^4+x+0.5
"""

def tau_base(x):
	Interval_Field=x.parent()
	tilde_x=x
	#if x is not in [-1,1]
	try:
		tilde_x=tilde_x.intersection(Interval_Field(-1,1))
	except ValueError:
		if (x<=-1):
			return Interval_Field(0)
		if (x>1):
			return Interval_Field(1)
	try:
		tilde_x_neg=tilde_x.intersection(Interval_Field(-1,0))
		val_neg = horner(x,1,3,Interval_Field(5)/2,0,0,1,0.5)
	except ValueError:
		val_neg=None
	try:
		tilde_x_pos=tilde_x.intersection(Interval_Field(0,1))
		val_pos = horner(x,-1,3,Interval_Field(-5)/2,0,0,1,0.5)
	except ValueError:
		val_pos=None
	if val_pos is None:
		return val_neg
	if val_neg is None:
		return val_pos
	return val_neg.union(val_pos)
	
def tau(x,i,partsize):
	x_tilde=x*partsize-i
	a=x.parent()(0)
	if (i==0):
		a=tau_base(x.parent()(0))/partsize
	return (tau_base(x_tilde)/partsize)-a

def tau_prime(x,i,partsize):
	return phi(x,i,partsize)

def tau_integral(i,partsize,prec = 53):
	RIF=RealIntervalField(prec = prec)
	if (i==0):
		return RIF(partsize-i)/(2*(partsize**2))-RIF(1)/(7*(partsize**2))
	if (i==partsize):
		return RIF(1)/(7*(partsize**2))
	if ((i>0) and (i<partsize)):
		return RIF(partsize-i)/(partsize**2)

def psi(x,i,partsize):
	a=0.5
	b=0.5
	if ((i==0) or (i==partsize)):
		a=1
		if (i==partsize):
			b=0
	if (i==partsize-1):
		b=1
	return a*tau(x,i,partsize)-b*tau(x,i+1,partsize)

def psi_prime(x,i,partsize):
	a=0.5
	b=0.5
	if ((i==0) or (i==partsize)):
		a=1
		if (i==partsize):
			b=0
	if (i==partsize-1):
		b=1
	return a*phi(x,i,partsize)-b*phi(x,i+1,partsize)

def psi_max(i,partsize, prec = 53):
	RIF=RealIntervalField(prec = prec)
	if ((i>1) and (i<(partsize-1))):
		x=RIF(1+2*i)/(2*partsize)
		return psi(x,i,partsize)

def psi_integral(i,partsize,prec = 53):
	a=0.5
	b=0.5
	if (i==0):
		a=1
	if (i==partsize-1):
		b=1
	return a*tau_integral(i,partsize,prec=prec)-b*tau_integral(i+1,partsize,prec=prec)


"""
We implement how a vector written in the basis
{psi,1} is plotted
"""

def return_basis(i,partsize): # for each i returns the element i/n, for i=n+1 returns the identity
	if i<partsize:
		return lambda x : psi(x,i,partsize)
	if i==partsize:
		return lambda x : tau(x,i,partsize)
	if i==partsize+1:
		return lambda x : (x.parent())(1) 

def return_basis_prime(i,partsize): # for each i returns the element i/n, for i=n+1 returns the identity
	if i<partsize:
		return lambda x : psi_prime(x,i,partsize)
	if i==partsize:
		return lambda x : tau_prime(x,i,partsize)
	if i==partsize+1:
		return lambda x : (x.parent())(0) 

	
def evaluate_vector_compact_basis_Liverani(x,v):
	n=len(v)
	
	partsize = (n-2)
	
	y=(x.parent())(v[partsize+1])
	
	x_tilde=x*partsize
	
	i_min=max(0,int(x_tilde.lower())-2)
	i_max=min(int(x_tilde.upper())+3,partsize+1)
	
	for i in range(i_min,i_max):
		f = return_basis(i,partsize)
		y+= v[i]*f(x)
	
	return y

def from_derivative_at_nodes_to_compact_liverani_basis(w):
	
	partsize=len(w)-2
	v=np.zeros(partsize+2)
	
	v[0]=w[0]
	
	for i in range(1,partsize): #cumulative sums for 1<=i<=n-1
		v[i]=v[i-1]+2*w[i]
	
	v[partsize] = v[partsize-1]+w[partsize]
	
	v[partsize+1]=w[partsize+1]
	
	return v

def compute_integral_liverani_basis(w, prec=53):
	
	n = len(w)-2
	
	riemann_sum=w[n+1]
	
	for i in range(0,n+1):
		riemann_sum+=w[i]*tau_integral(i,n,prec)

	return riemann_sum
	
def test_matrix(P):
	PP = sage_sparse_to_scipy_sparse(P.apply_map(lambda x: RR(x.center())))
	k=PP.shape[1]
	w=np.zeros(k)
	errors=np.zeros(k)
	for i in range(k):
		print i
		w[i]=1
		integral1=compute_integral_liverani_basis(w)
		print integral1
		w=prod_mat_vec(PP,w)
		integral2=compute_integral_liverani_basis(w)
		print integral2
		integral=integral1-integral2
		print (integral).absolute_diameter()/integral1.center()
		errors[i]=integral.center()
		
	return max(abs(errors))

def last_row(P):
	
	w = P.parent().row_space()(0)
	k=P.nrows()
	
	n=k-2
	
	Interval = (w[0]).parent()	
	
	w[0]=-1*tau_integral(0,n)
	
	for i in range(1,n):
		w[i]=-1*tau_integral(i,n)
	
	w[n]=-1*tau_integral(n,n)
	
	w[n+1]=Interval(1)
	
	v=P.linear_combination_of_rows(w)
	return v


def C1_norm_Liverani_basis(v):
	k=len(v)
	n=k-2
	
	Rup = RealField(rnd='RNDU')
	a=Rup(v[n+1])
	b=max(abs(v[0:n]))
	a+=3*b
	
	return a
		
def prod_mat_vec(PP,w):
	z=from_derivative_at_nodes_to_compact_liverani_basis(w)
	return PP*z

def interval_norminf_error(P):
	"""
	Compute the "norm-infinity width of the interval matrix P", i.e., an upper bound for :math:`\|P_1-P_2\|_inf`, where the :math:`P_i` are matrices inside the interval :math:`P`.
	
	This is essentially the norm of P.upper()-P.lower().
	"""
	
	w = VectorSpace(RealField(rnd='RNDU'),P.dimensions()[0])(0)
	for (ij, val) in P.dict().iteritems():
		w[ij[0]] += val.absolute_diameter() #operations are rounded up because the LHS is so
	return max(w)

def decay_time_single_vector_Liverani(PP, v, target, steperror):
	r"""
	Compute a ``N`` such that :math:`\|P^Nv\| \leq` ``target``.
	
	Args:
		P (numpy matrix):
		steperror (real): a value such that the error on the computed product :math:`Pv` is bounded by 
			``steperror*`` :math:`\|v\|`.
		my_norm (function): a function returning the (rigorous) norm to use.
	"""
	
	initialnorm = C1_norm_Liverani_basis(v) #this might be known a-priori or computed twice, but whatever
	Rup = RealField(rnd='RNDU') #default precision for this error computation; more shouldn't be needed
	error = Rup(0.0)
	for i in range(1000):
		v = prod_mat_vec(PP,v)
		currentnorm = C1_norm_Liverani_basis(v)
		error += steperror * currentnorm
		if error + currentnorm < target:
			return i+1
		if error > 0.5*target:
			raise ValueError, "The numerical error is growing too much; raise the precision."
		if currentnorm > 50*initialnorm: #nel caso infinito sta dando noia...
			raise ValueError, "The norm is growing too much; is this not contracting?"
		if i >= 999:
			raise ValueError, "Too many iterations; this looks strange."


def eigenerr_Liverani(P,target=0.5):
	
	PP=sage_sparse_to_scipy_sparse(P)
	
	k=PP.shape[0]
	n=k-2
	
	Rup = RealField(rnd='RNDU')
	
	#to take into account the fact that we are working with a basis
	#we have to contract much more the space to have an estimate 
	#of the norm
	target=Rup(target)/k
	
	Perror=3*interval_norminf_error(P)
	
	print Perror
	
	nnz = max_nonzeros_per_row(P)
	
	print "max NNZ per row ", nnz
	
	machineEpsilon = Rup(np.finfo(float).eps)
	gammak = machineEpsilon*nnz / (1-machineEpsilon*nnz)
	steperror = Rup(Perror) + Rup(gammak)
	
	print PP.shape[1]
	print steperror
	
	glob_dec_time=1
	
	for i in range(0,n+1):
		w=np.zeros(k)
		w[i]=1
		x=compute_integral_liverani_basis(w)
		w[n+1]=-1*x.center()
		
		dec_time=decay_time_single_vector_Liverani(PP, w, target, steperror)
		if (glob_dec_time<dec_time): glob_dec_time=dec_time
		if (i%1024==0):
			sys.stdout.write('Computing decay time:')
			sys.stdout.write("%d..." % i)
			sys.stdout.flush()
		
	return glob_dec_time

def estimate_norm_P_iterate_k(P,PP,l):
	
	Rup = RealField(rnd='RNDU')
	
	
	Perror=3*interval_norminf_error(P)
	
	print Perror
	
	nnz = max_nonzeros_per_row(P)
	
	print "max NNZ per row ", nnz
	
	machineEpsilon = Rup(np.finfo(float).eps)
	gammak = machineEpsilon*nnz / (1-machineEpsilon*nnz)
	steperror = Rup(Perror) + Rup(gammak)
	
	
	#PP=sage_sparse_to_scipy_sparse(P)
	k=PP.shape[0]
	n=k-2
	
	norm=Rup(0)
	error = Rup(0.0)
	
	for j in range(0,n+1):
		w=np.zeros(k)
		w[j]=1
		x=compute_integral_liverani_basis(w)
		w[n+1]=-1*x.center()
		for i in range(l):
			w = prod_mat_vec(PP,w)
			currentnorm = C1_norm_Liverani_basis(w)
			error += steperror * currentnorm
		if (error + currentnorm)*n > norm:
			norm=(error + currentnorm)*n
		if (j%128==0): print norm
	return norm
		
		
		
def compute_M(D):
	B=D.distorsion()
	lam=1/D.expansivity()
	return 1+B/(1-lam)
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
