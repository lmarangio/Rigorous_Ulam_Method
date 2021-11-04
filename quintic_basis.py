r"""
Implements a spline-like basis made of quintic functions.

The basis is composed of three kinds of functions: phi, psi, and tau, satisfying these relations:

:math:`\phi_i(x_j) = \delta_{ij}, \quad \phi'_i(x_j) = 0, \quad \phi''_i(x_j) = 0, \int_0^1 \phi_i(x) =1.`

:math:`\psi_i(x_j) = 0, \quad \psi'_i(x_j) = \delta_{ij}, \quad \phi''_i(x_j) = 0, \int_0^1 \psi_i(x) =0.`

:math:`\tau(x_j) = 0, \quad \tau'(x_j) = 0, \quad \tau''(x_j) = 0, \int_0^1 \tau(x) =1.`

The functions :math:`\phi_i(x)` are cusp functions, piecewise quintic. Their branches are given by :math:`\phi_-(x)=1+10*x^3+15*x^4+6*x^5` and :math:`phi_+(x)=1-10*x^3+15*x^4-6*x^5`.

:math:`\psi_i(x)` are TODO

:math:`\tau(x)` the sum of scaled piecewise quintics

This basis is implemented for **equispaced nodes** only.
"""

import numpy as np
from rigorous_integral import rigorous_integral_C1,rigorous_integral_C2
from partition import equispaced
from sage.all import *


__all__=["phi","phi_prime","psi","psi_prime","rigorous_integral_C1"]

def polyval(polynomial,x):
	"""
	Evaluates a polynomial p (given by its coefficients, e.g., [0, 1, 2] represents :math:`x+2x^2`) in an interval point x.
	Args:
		p (collection):
		x (number):
	Returns:
		:math:`p(x)`, evaluated using the operations of x.
	"""
	
	result = 0
	for coefficient in reversed(polynomial):
		result = x*result + coefficient
	
	return result

def phi(x,i,partsize):
	Interval_Field=x.parent()
	tilde_x=x*partsize-i
	#if x is not in [-1,1], return 0
	try:
		tilde_x=tilde_x.intersection(Interval_Field(-1,1))
	except ValueError:
		return Interval_Field(0)
	try:
		tilde_x_neg=tilde_x.intersection(Interval_Field(-1,0))
		val_neg = polyval([1,0,0,10,15,6], tilde_x_neg)
	except ValueError:
		val_neg=None
	try:
		tilde_x_pos=tilde_x.intersection(Interval_Field(0,1))
		val_pos = polyval([1,0,0,-10,15,-6], tilde_x_pos)
	except ValueError:
		val_pos=None
	if val_pos is None:
		return val_neg
	if val_neg is None:
		return val_pos
	return val_neg.union(val_pos)

def phi_prime(x,i,partsize):
	Interval_Field=x.parent()
	tilde_x=x*partsize-i
	#if x is not in [-1,1]
	try:
		tilde_x=tilde_x.intersection(Interval_Field(-1,1))
	except ValueError:
		return Interval_Field(0)
	try:
		tilde_x_neg=tilde_x.intersection(Interval_Field(-1,0))
		val_neg = polyval([0,0,30,60,30], tilde_x_neg)
	except ValueError:
		val_neg=None
	try:
		tilde_x_pos=tilde_x.intersection(Interval_Field(0,1))
		val_pos = polyval([0,0,-30,60,-30], tilde_x_pos)
	except ValueError:
		val_pos=None
	if val_pos is None:
		return val_neg
	if val_neg is None:
		return val_pos
	return val_neg.union(val_pos)

def phi_second(x,i,partsize):
	Interval_Field=x.parent()
	tilde_x=x*partsize-i
	#if x is not in [-1,1]
	try:
		tilde_x=tilde_x.intersection(Interval_Field(-1,1))
	except ValueError:
		return Interval_Field(0)
	try:
		tilde_x_neg=tilde_x.intersection(Interval_Field(-1,0))
		val_neg = polyval([0,60,180,120], tilde_x_neg)
	except ValueError:
		val_neg=None
	try:
		tilde_x_pos=tilde_x.intersection(Interval_Field(0,1))
		val_pos = polyval([0,-60,180,-120], tilde_x_pos)
	except ValueError:
		val_pos=None
	if val_pos is None:
		return val_neg
	if val_neg is None:
		return val_pos
	return val_neg.union(val_pos)

def psi(x,i,partsize):
	Interval_Field=x.parent()
	tilde_x=x*partsize-i
	rescale=1/Interval_Field(partsize)
	#if x is not in [-1,1]
	try:
		tilde_x=tilde_x.intersection(Interval_Field(-1,1))
	except ValueError:
		return Interval_Field(0)
	try:
		tilde_x_neg=tilde_x.intersection(Interval_Field(-1,0))
		val_neg = rescale * polyval([0,1,0,-6,-8,-3], tilde_x_neg)
	except ValueError:
		val_neg=None
	try:
		tilde_x_pos=tilde_x.intersection(Interval_Field(0,1))
		val_pos = rescale * polyval([0,1,0,-6,8,-3], tilde_x_pos)
	except ValueError:
		val_pos=None
	if val_pos is None:
		return val_neg
	if val_neg is None:
		return val_pos
	return val_neg.union(val_pos)

def psi_prime(x,i,partsize):
	Interval_Field=x.parent()
	tilde_x=x*partsize-i
	rescale=1/Interval_Field(partsize)
	#if x is not in [-1,1]
	try:
		tilde_x=tilde_x.intersection(Interval_Field(-1,1))
	except ValueError:
		return Interval_Field(0)
	try:
		tilde_x_neg=tilde_x.intersection(Interval_Field(-1,0))
		val_neg = rescale * polyval([1,0,-18,-32,-15], tilde_x_neg)
	except ValueError:
		val_neg=None
	try:
		tilde_x_pos=tilde_x.intersection(Interval_Field(0,1))
		val_pos = rescale * polyval([1,0,-18,32,-15], tilde_x_pos)
	except ValueError:
		val_pos=None
	if val_pos is None:
		return val_neg
	if val_neg is None:
		return val_pos
	return val_neg.union(val_pos)

def psi_second(x,i,partsize):
	Interval_Field=x.parent()
	tilde_x=x*partsize-i
	rescale=1/Interval_Field(partsize)
	#if x is not in [-1,1]
	try:
		tilde_x=tilde_x.intersection(Interval_Field(-1,1))
	except ValueError:
		return Interval_Field(0)
	try:
		tilde_x_neg=tilde_x.intersection(Interval_Field(-1,0))
		val_neg = rescale * polyval([0,-36,-96,-60], tilde_x_neg)
	except ValueError:
		val_neg=None
	try:
		tilde_x_pos=tilde_x.intersection(Interval_Field(0,1))
		val_pos = rescale * polyval([0,-36,96,-60],tilde_x_pos)
	except ValueError:
		val_pos=None
	if val_pos is None:
		return val_neg
	if val_neg is None:
		return val_pos
	return val_neg.union(val_pos)

def tau(x,partsize):
	n=2*partsize
	
	y = 0
	
	x_tilde=x*n
	
	i_min=int(x_tilde.lower())-2
	i_max=int(x_tilde.upper())+2
	
	if i_min<0: i_min=0
	if i_max>n: i_max=n
	
	for i in range(i_min,i_max):
		if ((i%2)==1):
			 y+=2*phi(x,i%n,n)
	return y

def tau_prime(x,partsize):
	n=2*partsize
	
	y = 0
	
	x_tilde=x*n
	
	i_min=int(x_tilde.lower())-2
	i_max=int(x_tilde.upper())+2
	
	if i_min<0: i_min=0
	if i_max>n: i_max=n
	
	for i in range(i_min,i_max):
		if ((i%2)==1):
			 y+=2*phi_prime(x,i%n,n)
	return y

def func_prod_dyn_unquotiented(f,f_prime,g,g_prime,dynamic):
	prod=lambda x : f(dynamic.f_unquotiented(x))*g(x)
	der_prod=lambda x: f_prime(dynamic.f_unquotiented(x))*dynamic.fprime(x)*g(x)+f(dynamic.f_unquotiented(x))*g_prime(x)
	return prod,der_prod

def func_prod_dyn(f,f_prime,g,g_prime,dynamic):
	prod=lambda x : f(dynamic.f(x))*g(x)
	der_prod=lambda x: f_prime(dynamic.f(x))*dynamic.fprime(x)*g(x)+f(dynamic.f(x))*g_prime(x)
	return prod,der_prod

def func_prod(f,f_prime,g,g_prime):
	prod=lambda x : f(x)*g(x)
	der_prod=lambda x: f_prime(x)*g(x)+f(x)*g_prime(x)
	return prod,der_prod

def func_prod_second(f,f_prime,f_second,g,g_prime,g_second):
	prod=lambda x : f(x)*g(x)
	der_prod=lambda x: f_prime(x)*g(x)+f(x)*g_prime(x)
	der_der_prod=lambda x: f_second(x)*g(x)+2*f_prime(x)*g_prime(x)+f(x)*g_second(x)
	return prod,der_prod,der_der_prod

def phi_circ_T(x,i,n,dynamic):
	Interval_Field=x.parent()
	y=Interval_Field(0)
	for j in range(dynamic.nbranches+1):
		y+=phi(dynamic.f_unquotiented(x),i+j*n,n)
	return y

def phi_circ_T_prime(x,i,n,dynamic):
	Interval_Field=x.parent()
	y=Interval_Field(0)
	for j in range(dynamic.nbranches+1):
		y+=phi_prime(dynamic.f_unquotiented(x),i+j*n,n)*dynamic.fprime(x)
	return y

def phi_circ_T_second(x,i,n,dynamic):
	Interval_Field=x.parent()
	y=Interval_Field(0)
	for j in range(dynamic.nbranches+1):
		y+=phi_second(dynamic.f_unquotiented(x),i+j*n,n)*(dynamic.fprime(x))**2+phi_prime(dynamic.f_unquotiented(x),i+j*n,n)*dynamic.fsecond(x)
	return y

def evaluate_spline(x,v):
	n=len(v)/2
	y=0
	x_tilde=x*n
	
	i_min=int(x_tilde.lower())-2
	i_max=int(x_tilde.upper())+2
	
	if i_min<0: i_min=0
	if i_max>n: i_max=n
	
	for i in range(i_min,i_max):
		y+=v[i%n]*phi(x,i%n,n)
		y+=v[i%n+n]*psi(x,i%n,n)
	
	y+=v[0]*phi(x-1,0,n)
	y+=v[n]*psi(x-1,0,n)
	return y

def evaluate_spline_prime(x,v):
	n=len(v)/2
	y=0
	x_tilde=x*n
	
	i_min=int(x_tilde.lower())-2
	i_max=int(x_tilde.upper())+2
	
	if i_min<0: i_min=0
	if i_max>n: i_max=n
	
	for i in range(i_min,i_max):
		y+=v[i%n]*phi_prime(x,i%n,n)
		y+=v[i%n+n]*psi_prime(x,i%n,n)
	return y

def evaluate_spline_corr(x,v):
	n=(len(v)-1)/2
	y=0
	x_tilde=x*n
	
	i_min=int(x_tilde.lower())-2
	i_max=int(x_tilde.upper())+2
	
	if i_min<0: i_min=0
	if i_max>n: i_max=n
	
	for i in range(i_min,i_max):
		y+=v[i%n]*phi(x,i%n,n)
		y+=v[i%n+n]*psi(x,i%n,n)
	
	y+=v[0]*phi(x-1,0,n)
	y+=v[n]*psi(x-1,0,n)
	
	y+=v[2*n]*tau(x,n)
	
	return y

def evaluate_spline_prime_corr(x,v):
	n=(len(v)-1)/2
	y=0
	x_tilde=x*n
	
	i_min=int(x_tilde.lower())-2
	i_max=int(x_tilde.upper())+2
	
	if i_min<0: i_min=0
	if i_max>n: i_max=n
	
	for i in range(i_min,i_max):
		y+=v[i%n]*phi_prime(x,i%n,n)
		y+=v[i%n+n]*psi_prime(x,i%n,n)
	
	y+=v[2*n]*tau_prime(x,n)
		
	return y


def evaluate_spline_C0(x,v):
	n=len(v)
	y=0
	x_tilde=x*n
	
	i_min=int(x_tilde.lower())-2
	i_max=int(x_tilde.upper())+2
	
	if i_min<0: i_min=0
	if i_max>n: i_max=n
	
	for i in range(i_min,i_max):
		y+=v[i]*phi(x,i,n)
		
	y+=v[0]*phi(x-1,0,n)
	return y

def evaluate_spline_prime_C0(x,v):
	n=len(v)
	y=0
	x_tilde=x*n
	
	i_min=int(x_tilde.lower())-2
	i_max=int(x_tilde.upper())+2
	
	if i_min<0: i_min=0
	if i_max>n: i_max=n
	
	for i in range(i_min,i_max):
		y+=v[i%n]*phi_prime(x,i%n,n)
	return y

def test_plot_vector(v):
	pl=plot(lambda x: evaluate_spline(RIF(x),v).center(), 0, 1, legend_label='phi')
	pl.show()

def test_plot_vector_C0(v):
	pl=plot(lambda x: evaluate_spline_C0(RIF(x),v).center(), 0, 1, legend_label='phi')
	pl.show()

def test_plot_vector_corr(v):
	pl=plot(lambda x: evaluate_spline_corr(RIF(x),v).center(), 0, 1, legend_label='phi')
	pl.show()

def components_phi(j,n,dynamic,epsilon=2**(-30)):
	v=np.zeros(2*n)
	width=2**(-35)
	partition=equispaced(n)
	for i in range(n):
		domain=RealInterval(j-1,j+1)/n
		f,f_prime=func_prod(lambda x:phi_circ_T(x,i,n,dynamic),lambda x:phi_circ_T_prime(x,i,n,dynamic),lambda x:phi(x,j,n),lambda x:phi_prime(x,j,n))
		v[i]+=(n*rigorous_integral_C1(f,f_prime,domain,width,iteration_count=0)).center()
	return v

def components_phi_C2(j,n,dynamic,epsilon=2**(-30)):
	v=np.zeros(2*n)
	width=2**(-35)
	partition=equispaced(n)
	for i in range(n):
		domain=RealInterval(j-1,j+1)/n
		v[i] += (n*rigorous_integral_C2(f,f_prime,f_second,domain,width,iteration_count=0)).center()
	return v

def test_sum_column(P,j,size):
	a=RealInterval(0)
	for i in range(size):
		a+=P[i,j]/size
	a+=P[2*size,j]
	return (a*size)
	
