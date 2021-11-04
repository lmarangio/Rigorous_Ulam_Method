from dynamic import *

_all__=["coeff_LY_C1_C0","coeff_LY_C2_C1","coeff_uniform_LY_C2_C1_complete"]


def coeff_LY_C1_C0(D,DD):
	lam_D=1/D.expansivity()
	lam_DD=1/DD.expansivity()
	
	
	B=D.distorsion()/(1-lam_D)
	M=D.bound_on_norms_of_powers('Linf')
	theta=lam_DD*M
	C=lam_DD*B+(1-lam_DD)*M
	return {'theta':theta, 'C':C}
	
def coeff_LY_C2_C1(D,DD):
	lam_D=1/D.expansivity()
	lam_DD=1/DD.expansivity()
	
	
	B=D.distorsion()/(1-lam_D)
	M=D.bound_on_norms_of_powers('Linf')
	theta=lam_DD*M
	C=lam_DD*B+(1-lam_DD)*M
	
	Z=1/(1-lam_D**2)*(D.third_distorsion()+lam_D/(1-lam_D)*B**2)
	
	mu=lam_DD*theta
	
	D=theta+C+3*max(1,B**2)*M+M*Z
	
	return {'mu':mu,'D':D}

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
