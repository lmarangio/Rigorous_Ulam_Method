"""
Files pertaining the (old) assembler using rigorously-computed integrals
"""

import numpy as np
from quintic_basis import phi,phi_prime,phi_second
from partition import equispaced
from dynamic import *
from rigorous_integral import almost_homogeneously_partition_interval

def computing_coefficient_phi_phi(i,j,n,domain,dynamic,step=5,width=2**(-30),iteration_count=0):
		iteration_count+=1
		Interval_Field=domain.parent()
		int_domains=almost_homogeneously_partition_interval(domain,step)
		
		riemann_sum=Interval_Field(0)
		
		for l in range(step-1):
			contrib_a=Interval_Field(0)
			a=Interval_Field(int_domains[l],int_domains[l+1])
			center_a=Interval_Field(a.center())
			radius_a=a.absolute_diameter()/2
			y=dynamic.f_unquotiented(center_a)
			y_a=dynamic.f_unquotiented(a)	
			l=((y*n).upper()).ceil()//n
			
			contrib_a+=2*phi(y,i+l*n,n)*phi(center_a,j,n)*radius_a
			dyn_prime_center_a=dynamic.fprime(center_a)
			dyn_prime_a=dynamic.fprime(a)
			
			#val_prime_center_a=phi_prime(y,i+l*n,n)*dyn_prime_center_a*phi(center_a,j,n)+phi(y,i+l*n,n)*phi_prime(center_a,j,n)
			val_second_center_a=phi_second(y,i+l*n,n)*(pow(dyn_prime_center_a,2))*phi(center_a,j,n)+phi_prime(y,i+l*n,n)*dynamic.fsecond(center_a)*phi(center_a,j,n)+2*phi_prime(y,i+l*n,n)*dynamic.fprime(center_a)*phi_prime(center_a,j,n)+phi(y,i+l*n,n)*phi_second(center_a,j,n)
			val_second_a=phi_second(y_a,i+l*n,n)*(pow(dyn_prime_a,2))*phi(a,j,n)+phi_prime(y_a,i+l*n,n)*dynamic.fsecond(a)*phi(a,j,n)+2*phi_prime(y_a,i+l*n,n)*dynamic.fprime(a)*phi_prime(a,j,n)+phi(y_a,i+l*n,n)*phi_second(a,j,n)
			contrib_a+=2*(val_second_center_a/3)*radius_a**3
			
			mag_error=(val_second_center_a-val_second_a).magnitude()
			error_a=Interval_Field(-mag_error,mag_error)*((radius_a)**3)/3
			contrib_a+=error_a
			if (contrib_a.absolute_diameter()>=width):
				riemann_sum+=computing_coefficient_phi_phi(i,j,n,a,dynamic,5,width,iteration_count)
			else: riemann_sum+=contrib_a
			
		return riemann_sum

def components_phi_C0(j,n,dynamic,epsilon=2**(-30)):
	v=np.zeros(2*n)
	width=2**(-35)
	partition=equispaced(n)
	for i in range(n):
		domain=RealInterval(j-1,j+1)/n
		v[i]+=(n*computing_coefficient_phi_phi(i,j,n,domain,dynamic)).center()
	return v

