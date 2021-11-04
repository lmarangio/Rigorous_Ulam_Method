from dynamic import *

def coeff_uniform_LY_C2_C1_complete(D,DD,partsize):
	
	lam_D=1/D.expansivity()
	lam_DD=1/DD.expansivity()
	
	
	B=D.distorsion()/(1-lam_D)
	M=D.bound_on_norms_of_powers('Linf')
	Z=1/(1-lam_D**2)*(D.third_distorsion()+lam_D/(1-lam_D)*B**2)
	
	
	tilde_mu=3*lam_DD**2*M+(M+3*lam_DD*B+3*B**2*M+M*Z)/partsize
	
	a=3*lam_DD*B+2*(3*B**2*M+M*Z)+M-tilde_mu
	b=3*lam_DD*M+3*lam_DD*B+6*lam_DD*B*M+2*(3*B**2*M+M*Z)+M-tilde_mu
	
	tilde_D=max(a,b)
	
	return {'tilde_mu':tilde_mu,'tilde_D':tilde_D, 'B':B,'M':M,'Z':Z,'lam_DD':lam_DD}
