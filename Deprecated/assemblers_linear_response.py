from basis_primitives import phi,eta,return_basis,return_basis_prime,psi_integral,tau_integral
from partition import equispaced, check_partition, is_iterable

from sage.rings.real_mpfr import RealField
from sage.all import RealNumber, NaN, floor
from sage.rings.real_mpfi import RealIntervalField, is_RealIntervalFieldElement
from sage.matrix.matrix_space import MatrixSpace
from joblib import Parallel, delayed
import sys

from sage.all import load
load('binsearch.spyx')

def assemble_on_branch_C2_pointwise_quintic_preserve_integral(dynamic, branch, partition, epsilon, prec = 53):
	"""
	Compute a matrix discretizing ``dynamic`` using piecewise quintic functions as basis.
	
	Assume that we are working on a torus, so there is no border and f(0)=f(1).
	
	See :fun:`assemble_piecewise_cubic_spline()` for details.
	"""
	
	n = len(partition)-1
	# we'll see later if something more optimized is needed
	
	F = RealIntervalField(prec=prec)
	M = MatrixSpace(F, n+2, n+2, sparse = True)
	P = M(0)
		
	sys.stdout.write("Assembling...")
	sys.stdout.flush()
	for i in range(n):
		
		y = dynamic.preimage(partition[i], branch, epsilon)
		Real_Interval=y.parent()
		
		Tprimey = dynamic.fprime(y)
		Tsecondy = dynamic.fsecond(y)
		# typically these two will be equal, unless the interval ``y`` contains a grid point.
		jmin = binsearch(y.lower(), partition)
		jmax = binsearch(y.upper(), partition)
		
		# phi_j(y) will be nonzero for j in [jmin, ..., jmax+1]
		# where jmax+1 may "overlap" to 0
		
		val=eta(y,n)/Tprimey
		P[i,n+1] += val
		P[n+1,n+1] -= val*integral_basis(i,n,prec)
			
		for j in range(jmin,jmax+2):
			
			# in each interval we have the partition of unity
			# we keep track of it and its derivative 
			# please note that, if we change this evaluation for the integrals,
			# we obtain a discretization that preserves integrals.
			val=phi(y, j, n) / Tprimey
			P[i,j] += val
			P[n+1,j] -= val*integral_basis(j,n,prec)
			
			if j>0 and j % 1024 == 0:
				sys.stdout.write("%d..." % j)
				sys.stdout.flush()
	
	if branch==0:
		P[n+1,n+1]+=1
		for j in range(n+1):
			#print(j)
			P[n+1,j]+=integral_basis(j,n,prec)
		
	sys.stdout.write("done.\n")
	sys.stdout.flush()
	return P

def assemble_on_branch_Liverani_basis_preserve_integral(dynamic, branch, partition, epsilon, prec = 53):
	"""
	"""
	
	n=len(partition)-1
	
	F = RealIntervalField(prec=prec)
	M = MatrixSpace(F, n+2, n+2, sparse = True)
	P = M(0)
		
	sys.stdout.write("Assembling...")
	sys.stdout.flush()
	
	if (branch==0): #we fill the values in  the last row with the values of the integrals
			
			for i in range(0,n):
				P[n+1,i]=psi_integral(i,n,prec)
			P[n+1,n]=tau_integral(n,n,prec)
			P[n+1,n+1] = F(1)
			
	for i in range(n+1): #we want to take also the last node, i.e, the (n)-th node is n/n=1
		
		y = dynamic.preimage(partition[i], branch, epsilon)
		Real_Interval=y.parent()
		
		Tprimey = dynamic.fprime(y)
		Tsecondy = dynamic.fsecond(y)
		# typically these two will be equal, unless the interval ``y`` contains a grid point.
		jmin = binsearch(y.lower(), partition)
		jmax = binsearch(y.upper(), partition)
		# phi_j(y) will be nonzero for j in [jmin-1, ..., jmax+2]
		
		
		jmin=max(0,jmin-1)
		jmax=min(jmax+3,n+1) #this is because the index j runs through the compact support elements... we treat the costant differently

		for j in range(jmin,jmax): #these for the compactly supported part of the basis 
			f = return_basis(j,n)
			f_prime = return_basis_prime(j,n)
			
			val = f_prime(y)/(Tprimey**2)-f(y)*Tsecondy /(Tprimey**3)
			P[i,j] += val
			"""
			if ((0<i) and (i<n)):
				P[n+1,j]-= val*(Real_Interval(n-i)/(n**2))
			elif (i==0):
				P[n+1,j]-= val*(Real_Interval(1)/(2*n)-Real_Interval(1)/(7*(n**2))) 		
			elif (i==n):
				P[n+1,j]-= val*(Real_Interval(1)/(7*(n**2)))
			"""
			if j>0 and j % 1024 == 0:
				sys.stdout.write("%d..." % j)
				sys.stdout.flush()
		
		j=n+1 	#now we compute this for the constant function 1
		f = return_basis(j,n) #j=n+1
		f_prime = return_basis_prime(j,n) #j=n+1
		val = f_prime(y)/(Tprimey**2)-f(y)*Tsecondy /(Tprimey**3)
		P[i,j] += val #j=n+1
		"""
		if ((0<i) and (i<n)):
			P[n+1,j]-= val*(Real_Interval(n-i)/(n**2))
		elif (i==0):
			P[n+1,j]-= val*(Real_Interval(1)/(2*n)-Real_Interval(1)/(7*(n**2))) 		
		elif (i==n):
			P[n+1,j]-= val*(Real_Interval(1)/(7*(n**2)))
		"""
	sys.stdout.write("done.\n")
	sys.stdout.flush()
	return P
