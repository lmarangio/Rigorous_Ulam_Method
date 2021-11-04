
from sage.all import *
from matrix_io import show_time
import numpy as np
import subprocess
import re
import time
import os
load('binsearch.spyx')

def apply_shift(vec, shift):
    if (shift == 0):
	return vec
    retv = VectorSpace(parent(vec), len(vec))(0)
    for i in range(len(vec)):
        j = i+shift if i + shift < len(vec) else 2*len(vec)-1-(i+shift)
        retv[j] += vec[i]
    return retv

# we assume meas_vec = approx of DENSITY
def variation_of_f(meas_vec):
    Rup = RealField(rnd='RNDU')
    var = Rup(0)

    for i in range(len(meas_vec)-1):
        var += abs(meas_vec[i]-meas_vec[i+1])

    return var

# the returned data is an list with
# * for each interval I
#   * for each preimage TiI:
#     - estremas of TiI,
#     - an upper bound for |1/T'| in TiI
#     - an upper bound for |T''/(T'^2)| in TiI
def preparation_for_estimate1(D, basis, epsilon):
    retv = []
    RI = D.field
    Rup = RealField(rnd='RNDU')
    Kest = len(basis)

    i = 0
    start_time = time.time()

    branch_limits = [[D.f(l) for l in bi] for bi in D.branch_intervals]

    for dual_element in basis.dual_composed_with_dynamic(D, epsilon):
        I = RI(i,i+1)/Kest
        preims = []

        bnum = 0
        for a,b in dual_element: #iterate over the preimages
	    A = a.min(b)
	    B = a.max(b)

            TiI = RI(A,B)
            invtp_ab = abs(1/D.f_prime(TiI)).upper()
            distortion_ab = abs(D.f_second(TiI) / (D.f_prime(RI(A,B))**2)).upper()

            bterm = None
            for bl in branch_limits[bnum]:
                if I.overlaps(bl):
                    if bterm == None:
                        bterm = []
                    bterm.append( (bl, abs(1/D.f_prime(bl)).upper()) )
            
            preims.append( (RR(A),RR(B),Rup(invtp_ab),Rup(distortion_ab), bterm) )
            bnum += 1
        retv.append(preims)

        i += 1
        if (i % 256 == 0):
            elapsed = time.time() - start_time
            to_completion = elapsed / i * (len(basis)-i)
       	    print ("%d (%.02f%%, elapsed: %s, ETA: %s)" % (i,
                   RealField(prec=53)(i*100.0/len(basis)),
                   show_time(elapsed),
                   show_time(to_completion)
            ) )
    return retv

# we assume meas_vec = approx of DENSITY function
#
# Notice that N(1-pi)Lf is always < delta/2 Var(rho) by L^1/W estimate.
#
# Here we progressively estimate the variation of Lf to use the
# stronger BV/W estimate. More precisely, for each interval I we can take
# the minimum between
#  - (variation in I) * (delta/4)
#  - (L^1 norm in I)
# Taking the sum (and multiplying by delta/2 Var(rho)) we have the estimate.
# If we where just considering the L^1 norm we recover the dummy estimate.
#
# Other returned values are the estimations of
# * bound for the measure of Lf_delta
# * bound for the variation of Lf_delta
# (in the (coarser) partition).
#
def estimate_W_norm_of_1mpi_Lf(prep, density_vec, density_basis):
    Rup = RealField(rnd='RNDU')
    Raway = RealField(rnd='RNDA')

    Kest = len(prep)
    delta = Rup(1.0) / Rup(len(density_vec))
    estimate = Rup(0)
    trivial_est = Rup(0)
    
    start_time = time.time()
    
    meas_Lf = VectorSpace(Rup, Kest)(0)
    var_Lf = VectorSpace(Rup, Kest)(0)

    i = 0
    for preims in prep:
        nnz = 0
        meas_in_Ii = 0
        var_in_Ii = 0

        branch_num = 0
        for A,B,invtp_ab,distortion_ab, bterm in preims: #iterate over the preimages

            meas_in_ab = Rup(0)
            var_in_ab = Rup(0)

            jmin = binsearch(A, density_basis.partition)
	    jmax = binsearch(B, density_basis.partition)

	    for j in range(jmin, jmax+1):
		# ensure that they are sorted
				
		x = density_basis.partition[j]
		y = density_basis.partition[j+1]

		# compute endpoints of the intersection
		lower = Rup(max(A,x))
		upper = Rup(min(B,y))
		proportion = Rup(max(upper - lower, 0)) / Rup(y-x)

                # essentially here we compute L_delta * f_delta
                meas_in_ab += proportion * Rup(density_vec[j]) * delta
                if(j<jmax):
                    var_in_ab += Rup(abs(Raway(density_vec[j]) - Raway(density_vec[j+1])))

                if var_in_ab.is_NaN():
                    print 'density_vec[j]=',density_vec[j]
                    print 'density_vec[j+1]=',density_vec[j+1]
                    raise ValueError, 'var_in_an is Nan!'

            # this estimates the variation coming from this branch ab
            estimate_var_Ii_from_ab = invtp_ab * var_in_ab + distortion_ab * meas_in_ab

            if var_in_ab.is_zero() and meas_in_ab.is_zero():
               estimate_var_Ii_from_ab = Rup(0)                
            elif (invtp_ab.is_positive_infinity() or
               distortion_ab.is_positive_infinity()):
               estimate_var_Ii_from_ab = Rup('+infinity')
               
            if estimate_var_Ii_from_ab.is_NaN():
                print 'invtp_ab=',invtp_ab
                print 'var_in_ab=',var_in_ab
                print 'distortion_ab=',distortion_ab
                print 'meas_in_ab=',meas_in_ab
                raise ValueError, 'estimate_var_Ii_from_ab is Nan!'

            # boundary terms
            if bterm:
                for bl, b_invtp in bterm:
                    jmin = binsearch(bl.lower(), density_basis.partition)
	            jmax = binsearch(bl.upper(), density_basis.partition)
	            for j in range(jmin, jmax+1):
                        estimate_var_Ii_from_ab += Rup(density_vec[j])*b_invtp

                
            
            meas_in_Ii += meas_in_ab
            var_in_Ii += estimate_var_Ii_from_ab

            # this is the other term that we can use, and estimate the minimum
            alt_est = (delta/4) * estimate_var_Ii_from_ab
            estimate += min(alt_est, meas_in_ab)
            trivial_est += meas_in_ab
            
            branch_num += 1

        if meas_in_Ii.is_NaN():
            raise ValueError, 'meas_in_Ii is NaN!'
        if var_in_Ii.is_NaN():
            raise ValueError, 'var_in_Ii is NaN!'
        meas_Lf[i] = meas_in_Ii
        var_Lf[i] = var_in_Ii
        i += 1

        if (i % 256 == 0):
            elapsed = time.time() - start_time
            to_completion = elapsed / i * (Kest-i)
       	    print ("%d (%.02f%%, elapsed: %s, ETA: %s) " % (i,
                   RealField(prec=53)(i*100.0/Kest),
                   show_time(elapsed),
                   show_time(to_completion)
            ) )
            print "%s (of %s)" % (str(estimate), str(trivial_est))

    return (estimate * delta/2), meas_Lf, var_Lf


def reflect(i, size):
    if i<0:
        return 1-i
    if i>=size:
        return 2*size - i - 1
    return i

def mod(i, size):
    return (i+size) % size


def estimate_var_of_NLf(meas_Lf, var_Lf, rel_noise_size, mod1):
    Rup = RealField(rnd='RNDU')
    
    Kest = len(meas_Lf)
    var_NLf = VectorSpace(Rup, Kest)(0)

    #abs_noise_size = int(rel_noise_size * K)
    abs_xiq2f = int(floor(rel_noise_size * Kest / 2))
    abs_xiq2c = int(ceil(rel_noise_size * Kest / 2))

    boundary = mod if mod1 else reflect
    for i in range(Kest):
        est1 = (meas_Lf[boundary(i-abs_xiq2c, Kest)] + \
               meas_Lf[boundary(i+abs_xiq2c, Kest)] + \
               meas_Lf[boundary(i-abs_xiq2f, Kest)] + \
               meas_Lf[boundary(i+abs_xiq2f, Kest)]) / rel_noise_size

        est2 = sum(var_Lf[boundary(j, Kest)] for j in range(i-abs_xiq2c,i+abs_xiq2c+1)) \
               / (2 * abs_xiq2c)
        var_NLf[i] = min(est1, est2)
        if var_NLf[i].is_NaN():
            raise ValueError, 'var_NLf[i] is NaN!'

    #if any(x.is_NaN() for x in var_NLf):
    #    raise ValueError, 'some of var_NLf[i] is NaN!'
    #if sum(var_NLf).is_NaN():
    #    raise ValueError, 'some of var_NLf[i] is NaN!'

    return var_NLf

def estimate_meas_of_NLf(meas_Lf, rel_noise_size, mod1):
    Rup = RealField(rnd='RNDU')
    
    K = len(meas_Lf)
    meas_NLf = VectorSpace(Rup, K)(0)

    abs_noise_size = int(rel_noise_size * K)
    abs_xiq2 = abs_noise_size / 2

    boundary = mod if mod1 else reflect
    for i in range(K):
        meas_NLf[i] = sum(meas_Lf[boundary(j, K)] for j in range(i-abs_xiq2,i+abs_xiq2+1)) \
               / abs_noise_size

    return meas_NLf


# Computes a bound for the derivative (that will be used as an
# estimation of the W->W norm) for all interval in the partition.
def preparation_for_estimate2(D, basis):
    Rup = RealField(rnd='RNDU')
    
    K = len(basis)
    derivs = VectorSpace(Rup, K)(0)

    i = 0
    start_time = time.time()

    for i in range(K):
        I = D.field(basis.partition[i], basis.partition[i+1])
        derivs[i] = Rup(abs(D.f_prime(I)).upper())
    return derivs

#
# returns an estimate of the L1-norm of NL(1-pi)NLf_\delta
#
# For each interval, this is done taking the minimum of
#  - variation * (Var->W norm of 1-pi) * (W norm of L) * (W->L1 norm of N)
#  - variation * (Var->L1 norm of 1-pi)
# If we where only considering the second, we have the dummy estimate
#
def estimate_L1_norm_of_NL_1mpi_NLf(var_NLf, est_deriv, delta, varRho):
    Rup = RealField(rnd='RNDU')

    retv = Rup(0)
    K = len(var_NLf)

    for i in range(K):
        retv += min(delta*varRho/4*est_deriv[i], 1) * var_NLf[i]

    return retv * (delta/2)


def estimate_L1_error_apriori(K, noise_size_rel, alpha, sumCi):
    Rup = RealField(rnd='RNDU')
    delta = Rup(1)/K
    varRho = 2/noise_size_rel
    return (1/(1-alpha)) * (2*sumCi + 1) * (delta/2) * varRho


def estimate_L1_error_aposteriori(prep1, prep2,
                                  meas_vec, numeric_error, meas_basis,
                                  noise_size_rel, alpha, sumCi,
                                  shift, mod1):
    Rup = RealField(rnd='RNDU')
    K = len(meas_vec)
    delta = Rup(1)/K
    varRho = 2/noise_size_rel

    apriori_error = estimate_L1_error_apriori(K, noise_size_rel, alpha, sumCi)
    apriori_error += numeric_error
    
    print 'Performing the estimate of type 1'
    est1, meas_Lf, var_Lf = estimate_W_norm_of_1mpi_Lf(prep1, meas_vec, meas_basis)
    meas_Lf = apply_shift(meas_Lf, shift)
    var_Lf = apply_shift(var_Lf, shift)
    
    print 'Estimate var_NLf'
    var_NLf = estimate_var_of_NLf(meas_Lf, var_Lf, noise_size_rel, mod1)
    tot_var_NLf = sum(var_NLf)
    print 'tot_var_NLf=',tot_var_NLf

    #joblib.dump(meas_Lf, 'cz_meas_Lf')
    #joblib.dump(var_Lf, 'cz_var_Lf')
    #joblib.dump(var_NLf, 'cz_var_NLf')

    # elements of the first type (1-pi)f, that we estimate as (1-pi)NLf_delta + ...
    A1 = delta/2*tot_var_NLf
    B1 = delta/2*varRho
    print 'A1=',A1,'B1=',B1

    # estimate the second amount of elements of a certain type
    A2 = varRho * est1
    B2 = delta/2*varRho
    print 'A2=',A2,'B2=',B2

    if any(x.is_NaN() for x in [A1,B1,A2,B2]):
        raise ValueError, 'got a Nan from type-1 estimate?'

    # elements of the third type
    print 'Performing the estimate of type 2'
    # the shift here, if any, is not relevant (the W norm is contracted)
    est2 = estimate_L1_norm_of_NL_1mpi_NLf(var_NLf, prep2, delta, varRho)
    A3 =  est2 + (delta**2 / 4 * varRho * tot_var_NLf)
    B3 = delta/2*varRho
    print 'A3=',A3,'B3=',B3

    if any(x.is_NaN() for x in [A3,B3]):
        raise ValueError, 'got a Nan from type-2 estimate?'

    A = A1 + sumCi*(A2+A3)
    B = B1 + sumCi*(B2+B3)
    C = A/(1-alpha)
    D = B/(1-alpha)
    L1ErrorBound = (C + numeric_error)/(1-D)
    print 'alpha=',alpha
    print 'A=',A,'B=',B
    print 'C=',C,'D=',D
    print 'L1ErrBound=',L1ErrorBound

    if any(x.is_NaN() for x in [A,B,C,D,L1ErrorBound]):
        raise ValueError, 'got a Nan from type-2 estimate?'

    meas_NLf = estimate_meas_of_NLf(meas_Lf, noise_size_rel, mod1)

    Linf_to_real_f = L1ErrorBound / noise_size_rel
    Kest = len(meas_NLf)
    Linf_estimate = VectorSpace(Rup, Kest)(0)
    for i in range(Kest):
        Linf_estimate[i] = var_NLf[i] + meas_NLf[i]*Kest + Linf_to_real_f

    return L1ErrorBound, apriori_error, Linf_estimate
    
