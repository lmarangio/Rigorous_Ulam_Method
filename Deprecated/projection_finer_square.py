"""
	Implements the elements of the partition of unity
"""

"""
The branches are
phi-(x)=1+10*x^3+15*x^4+6*x^5
phi+(x)=1-10*x^3+15*x^4-6*x^5
"""
def phi(x,i,partsize):
	Interval_Field=x.parent()
	tilde_x=x*partsize-i
	#if x is not in [-1,1]
	try:
		tilde_x=tilde_x.intersection(Interval_Field(-1,1))
	except ValueError:
		return Interval_Field(0)
	try:
		tilde_x_neg=tilde_x.intersection(Interval_Field(-1,0))
		val_neg=Interval_Field(1)+10*pow(tilde_x_neg,3)+15*pow(tilde_x_neg,4)+6*pow(tilde_x_neg,5)
	except ValueError:
		val_neg=None
	try:
		tilde_x_pos=tilde_x.intersection(Interval_Field(0,1))
		val_pos=Interval_Field(1)-10*pow(tilde_x_pos,3)+15*pow(tilde_x_pos,4)-6*pow(tilde_x_pos,5)
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
		val_neg=(30*pow(tilde_x_neg,2)+60*pow(tilde_x_neg,3)+30*pow(tilde_x_neg,4))*partsize
	except ValueError:
		val_neg=None
	try:
		tilde_x_pos=tilde_x.intersection(Interval_Field(0,1))
		val_pos=(-30*pow(tilde_x_pos,2)+60*pow(tilde_x_pos,3)-30*pow(tilde_x_pos,4))*partsize
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
		val_neg=(60*tilde_x_neg+180*pow(tilde_x_neg,2)+120*pow(tilde_x_neg,3))*(partsize**2)
	except ValueError:
		val_neg=None
	try:
		tilde_x_pos=tilde_x.intersection(Interval_Field(0,1))
		val_pos=(-60*tilde_x_pos+180*pow(tilde_x_pos,2)-120*pow(tilde_x_pos,3))*(partsize**2)
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
		val_neg=(tilde_x_neg-6*pow(tilde_x_neg,3)-8*pow(tilde_x_neg,4)-3*pow(tilde_x_neg,5))*rescale
	except ValueError:
		val_neg=None
	try:
		tilde_x_pos=tilde_x.intersection(Interval_Field(0,1))
		val_pos=(tilde_x_pos-6*pow(tilde_x_pos,3)+8*pow(tilde_x_pos,4)-3*pow(tilde_x_pos,5))*rescale
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
	#if x is not in [-1,1]
	try:
		tilde_x=tilde_x.intersection(Interval_Field(-1,1))
	except ValueError:
		return Interval_Field(0)
	try:
		tilde_x_neg=tilde_x.intersection(Interval_Field(-1,0))
		val_neg=(1-18*pow(tilde_x_neg,2)-32*pow(tilde_x_neg,3)-15*pow(tilde_x_neg,4))
	except ValueError:
		val_neg=None
	try:
		tilde_x_pos=tilde_x.intersection(Interval_Field(0,1))
		val_pos=(1-18*pow(tilde_x_pos,2)+32*pow(tilde_x_pos,3)-15*pow(tilde_x_pos,4))
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
	#if x is not in [-1,1]
	try:
		tilde_x=tilde_x.intersection(Interval_Field(-1,1))
	except ValueError:
		return Interval_Field(0)
	try:
		tilde_x_neg=tilde_x.intersection(Interval_Field(-1,0))
		val_neg=(-36*tilde_x_neg-96*pow(tilde_x_neg,2)-60*pow(tilde_x_neg,3))*partsize
	except ValueError:
		val_neg=None
	try:
		tilde_x_pos=tilde_x.intersection(Interval_Field(0,1))
		val_pos=(-36*tilde_x_pos+96*pow(tilde_x_pos,2)-60*pow(tilde_x_pos,3))*partsize
	except ValueError:
		val_pos=None
	if val_pos is None:
		return val_neg
	if val_neg is None:
		return val_pos
	return val_neg.union(val_pos)

def evaluate_spline_C0_phi(x,v,n):
	
	a=v[0:n] #extracts the first part of the vector, where the information about the phi_i is kept
	
	y=0
	
	x_tilde=x*n # we do not want to evaluate all the phi_i, to fasten up computations
	
	i_min=int(x_tilde.lower())-2
	i_max=int(x_tilde.upper())+2
	
	if i_min<0: i_min=0
	if i_max>n: i_max=n
	
	for i in range(i_min,i_max):
		y+=a[i]*phi(x,i,n)
		
	y+=v[0]*phi(x-1,0,n)
	return y
	
def evaluate_spline_C0_psi(x,v,n):
	
	a=v[n:n+n**2] #extracts the first part of the vector, where the information about the psi_i is kept
	
	y=0
	
	x_tilde=x*(n**2) # we do not want to evaluate all the phi_i, to fasten up computations, so we find which are the non zero psi
	
	i_min=int(x_tilde.lower())-2
	i_max=int(x_tilde.upper())+2
	
	if i_min<0: i_min=0
	if i_max>n: i_max=n**2
	
	for i in range(i_min,i_max):
		y+=a[i]*psi(x,i,n**2)
		
	y+=a[0]*psi(x-1,0,n**2)
	return y

def evaluate_spline_finer_derivative(x,v,n):
	y1=evaluate_spline_C0_phi(x,v,n)
	y2=evaluate_spline_C0_psi(x,v,n)
	return y1+y2
