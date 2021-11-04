
from sage.all import sqrt


def rho(a,b,c,d):	
	return 0.5*(a+d+sqrt((a-d)**2+4*b*c))

def autonorm(a,b,c,d):
	v_1=a-d+sqrt((a-d)**2+4*b*c)
	v_2=2*b
	lam=v_1+v_2
	
	return [v_1/lam,v_2/lam]
