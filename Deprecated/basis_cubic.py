from sage.rings.real_mpfr import RealField
from sage.all import RealNumber, NaN, floor
from sage.rings.real_mpfi import RealIntervalField, is_RealIntervalFieldElement

import numpy as np


import sys



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
phi-(x)=1-3x^2-2x^3
phi+(x)=1-3x^2+2x^3
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
		val_neg = horner(x,-2,-3,0,1)
	except ValueError:
		val_neg=None
	try:
		tilde_x_pos=tilde_x.intersection(Interval_Field(0,1))
		val_pos = horner(x,2,-3,0,1)
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

def integral_phi_basis(i,n,prec = 53):
	RIF=RealIntervalField(prec)
	if (i==0) or (i==n):
		 return RIF(1)/(2*n)
	elif (i==n+1):
		return RIF(1)
	else :
		return RIF(1)/n

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
		val_neg = (horner(x,-6,-6,0))
	except ValueError:
		val_neg=None
	try:
		tilde_x_pos=tilde_x.intersection(Interval_Field(0,1))
		val_pos = (horner(x,6,-6,0))
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
0.5+x-x^3-x^4/2
0.5+x-x^3+x^4/2
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
		val_neg = horner(x,-0.5,-1,0,1,0.5)
	except ValueError:
		val_neg=None
	try:
		tilde_x_pos=tilde_x.intersection(Interval_Field(0,1))
		val_pos = horner(x,0.5,-1,0,1,0.5)
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
		return RIF(7)/(20*(partsize**2))+RIF(partsize-1)/(2*(partsize**2)) 
	if (i==partsize):
		return (RIF(1)/2-RIF(7)/20)*1/(partsize**2)
	if ((i>0) and (i<partsize)):
		return  RIF(partsize-i)/(partsize**2)

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

"""
Some functions that returns constants of the basis
"""
###Check!!!
def psi_max(i,partsize, prec = 53):
	RIF=RealIntervalField(prec = prec)
	if ((i>1) and (i<(partsize-1))):
		x=RIF(1+2*i)/(2*partsize)
		return psi(x,i,partsize)

def psi_integral(i,partsize,prec = 53):
	a=0.5
	b=0.5
	if ((i==0) or (i==partsize)):
		a=1
		if (i==partsize):
			b=0
	if (i==partsize-1):
		b=1
	return a*tau_integral(i,partsize,prec=prec)-b*tau_integral(i+1,partsize,prec=prec)


"""
We implement how a vector written in the basis
{psi,1} is plotted
"""

def return_basis(i,partsize): # for each i returns the element i/n, for i=n+1 returns the identity
	if i<partsize+1:
		return lambda x : psi(x,i,partsize)
	if i==partsize+1:
		return lambda x : (x.parent())(1) 

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

