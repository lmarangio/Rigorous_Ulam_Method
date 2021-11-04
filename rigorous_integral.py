import numpy as np

def almost_homogeneously_partition_interval(domain,n):
	return np.linspace(domain.lower(),domain.upper(),n)

#rigorous integration method of f\circ T g
def rigorous_integral_C1(f,f_prime,domain,width=2**(-30),iteration_count=0):
	Interval_Field=domain.parent()
	
	iteration_count+=1
	#print(iteration_count)
	int_domains=almost_homogeneously_partition_interval(domain,32)
	riemann_sum=Interval_Field(0)
	
	for i in range(31):
		a=Interval_Field(int_domains[i],int_domains[i+1])
		#print(a.endpoints())
		center_a=a.center()
		radius_a=a.absolute_diameter()/2
		contrib_a=2*f(Interval_Field(center_a))*radius_a
		#print(f_prime(a))
		mag_error=(f_prime(a)-f_prime(Interval_Field(center_a))).magnitude()
		error_a=Interval_Field(-mag_error,mag_error)/2*((radius_a)**2)
		contrib_a+=error_a
		if (contrib_a.absolute_diameter()>=width):
			riemann_sum+=rigorous_integral_C1(f,f_prime,a,width,iteration_count)
		else: riemann_sum+=contrib_a
	return riemann_sum

def rigorous_integral_C2(f,f_prime,f_second,domain,width=2**(-30),iteration_count=0):
	Interval_Field=domain.parent()
	partition_size=4
	iteration_count+=1
	#print(iteration_count)
	int_domains=almost_homogeneously_partition_interval(domain,partition_size)
	riemann_sum=Interval_Field(0)
	
	for i in range(partition_size-1):
		a=Interval_Field(int_domains[i],int_domains[i+1])
		#print(a.endpoints())
		center_a=a.center()
		radius_a=a.absolute_diameter()/2
		contrib_a=2*(f(Interval_Field(center_a))*radius_a+(f_second(Interval_Field(center_a))/3)*(radius_a)**3)
		#print(f_prime(a))
		mag_error=(f_second(a)-f_second(Interval_Field(center_a))).magnitude()
		error_a=Interval_Field(-mag_error,mag_error)*((radius_a)**3)/3
		contrib_a+=error_a
		if (contrib_a.absolute_diameter()>=width):
			riemann_sum+=rigorous_integral_C2(f,f_prime,f_second,a,width,iteration_count)
		else: riemann_sum+=contrib_a
	return riemann_sum

