
from sage.all import *
from matrix_io import *
import numpy as np
import subprocess
import re
import time
import os
import joblib
from noise_gpu_settings import *
load('binsearch.spyx')

#def dot_prod(a, b):
#    RI53 = RealIntervalField(53)
#    l = len(a)
#    return sum(a[i] * b[i] for i in range(l)) / l

def dot_prod_notin(a, b, forbidden):
    RI53 = RealIntervalField(53)
    l = len(a)
    return sum(a[i] * RI53(b[i][0], b[i][1]) \
               for i in range(l) if not i in forbidden) / l

def dot_prod_in_interval(a, b, RI, interval):
    return sum(a[i] * RI(b[i][0], b[i][1]) \
               for i in range(*interval)) / len(a)

def mass_in_interval(a, RI, interval):
    return sum(RI(a[i]) for i in range(*interval)) / len(a)

def union_in_interval(b, RI, interval):
    return RI( min(b[i][0] for i in range(*interval)),
               max(b[i][1] for i in range(*interval)) )

def max_in_interval(a, interval):
    return max(a[i] for i in range(*interval))

def compute_component_at_s1(D, K,
                            meas_vec, # approximation of the measure
                            measLinfinity_est,  # L-infinity norm of the measure
                            # the element of the partition s.t.
                            # [idx1/K, (idx1+2)/K] contain the singularity
                            idx1,
                            # boundaries around the singularity s1
                            s1_boundary1, s1_boundary2):
    RI = D.field
    lyap = D._0
    logdf_LInfEst = D._0

    # S1: meas_L1 * logf'_Linf estimate in the first part
    logdf = log(abs(D.f1_prime(RI(idx1 / K, s1_boundary1.upper()))))
    add = logdf * RI((meas_vec[idx1] + meas_vec[idx1+1]) / K).union(RI(0))
    lyap += add
    
    print "..A1: [%s %s]\n" % (str(add.lower()), str(add.upper()))
    logdf_LInfEst = logdf_LInfEst.max(logdf.abs())

    # S1: meas_Linf * logf'_L1 estimate at the singularity
    # as meas is positive, and logdf does not changes it sign,
    # the true value is between the result and zero
    #logdf_norm_L1 = D.bound_int_p_q_log_f1_prime(s1_boundary1, s1_boundary2)
    logdf_norm_L1 = D.bound_int_p_sing1_log_f1_prime(s1_boundary1) + \
                    D.bound_int_sing1_q_log_f1_prime(s1_boundary2)
    add = (logdf_norm_L1 * measLinfinity_est).union(RI(0))
    lyap += add
    print "A1A2: [%s %s]\n" % (str(add.lower()),str(add.upper()))

    # S1: meas_L1 * logf'_Linf estimate in the last part
    logdf = log(abs(D.f1_prime(RI(s1_boundary2.lower(), (idx1+2) / K))))
    add = logdf * RI((meas_vec[idx1] + meas_vec[idx1+1]) / K).union(RI(0))
    lyap += add
    print "A2..: [%s %s]\n" % (str(add.lower()),str(add.upper()))
    logdf_LInfEst = logdf_LInfEst.max(logdf.abs())

    return lyap, logdf_LInfEst


def compute_component_at_s2(D, K,
                            meas_vec, # approximation of the measure
                            measLinfinity_est,  # L-infinity norm of the measure
                            # the element of the partition s.t.
                            # [idx2/K, (idx2+2)/K] contain the singularity
                            idx2,
                            # boundaries around the singularity s1
                            s2_boundary1, s2_boundary2):
    RI = D.field
    lyap = D._0
    logdf_LInfEst = D._0
    
    # S2: meas_L1 * logf'_Linf estimate in the first part
    logdf = log(abs(D.f1_prime(RI(idx2 / K, s2_boundary1.upper()))))
    add = logdf * RI((meas_vec[idx2] + meas_vec[idx2+1]) / K).union(RI(0))
    lyap += add
    print "..B1: [%s %s]\n" % (str(add.lower()),str(add.upper()))
    iv = RI(idx2 / K, s2_boundary1.upper())
    print "iv:", iv.lower(), iv.upper()
    print "ldf:", logdf.lower(), logdf.upper()
    logdf_LInfEst = logdf_LInfEst.max(logdf.abs())
    print "ldfE:", logdf_LInfEst.lower(), logdf_LInfEst.upper()

    logdf_norm_L1 = D.bound_int_p_fence_log_f1_prime(s2_boundary1)
    add = (logdf_norm_L1 * measLinfinity_est).union(RI(0))
    lyap += add
    print "B1.B: [%s %s]\n" % (str(add.lower()),str(add.upper()))

    # S2: meas_Linf * logf'_L1 estimate at the singularity
    # as meas is positive, and logdf does not changes it sign,
    # the true value is between the result and zero
    logdf_norm_L1 = D.bound_int_fence_q_log_f2_prime(s2_boundary2)
    add = (logdf_norm_L1 * measLinfinity_est).union(RI(0))
    lyap += add
    print "B.B2: [%s %s]\n" % (str(add.lower()),str(add.upper()))

    # S2: meas_L1 * logf'_Linf estimate in the last part
    logdf = D.f2_prime(RI(s2_boundary2.lower(), (idx2 + 2) / K)).abs().log()
    add = logdf * RI((meas_vec[idx2] + meas_vec[idx2+1]) / K).union(RI(0))
    lyap += add
    print "B2..: [%s %s]\n" % (str(add.lower()),str(add.upper()))
    logdf_LInfEst = logdf_LInfEst.max(logdf.abs())

    return lyap, logdf_LInfEst


COMPINVMEAS_OPENCL_PATH = '/home/gigi/Desktop/LuigiWork/compinvmeas-opencl'

#def compute_exp(meas_vec, meas_file,
#                measureL1_Err, K, D,
#                deriv_vec, deriv_file,
#                noise_size_rel):
#    RI = D.field
#
#    # get the positions of the singularities
#    singul1 = (D._0d125 - D._start) * D._iscale
#    singul2 = (D._fence - D._start) * D._iscale
#
#    # grid points around the singularities, and select the forbidden region
#    idx1 = Integer(floor(singul1 * K - 0.5).lower())
#    idx2 = Integer(floor(singul2 * K - 0.5).lower())
#    forbidden = [idx1, idx1+1, idx2, idx2+1]
#    ranges = [[0, idx1], [idx1+2, idx2], [idx2+2, K]]
#
#    #print 'Lyapunov, computing max'
#    #base_logdf_LInfEst = max(max(abs(deriv_vec[i][0]), abs(deriv_vec[i][1])) \
#    #                          for i in range(K) if not i in forbidden)
#    #base_logdf_LInfEst = RI(base_logdf_LInfEst)
#    #print 'Linf of log(df) in the region is ',base_logdf_LInfEst.upper()
#    #
#    # compute lyap and max log f', simple part far from singularities
#    #print 'Lyapunov, computing base component'
#    #base_lyap = dot_prod_notin(meas_vec, deriv_vec, forbidden)
#
#    ranges_spec = ','.join('-'.join(str(i) for i in x) for x in ranges)
#    out_file = 'out.dat'
#    args = ['./ComputeRangeDotProduct',
#            str(K),
#            os.path.abspath(meas_file),
#            os.path.abspath(deriv_file),
#            ranges_spec,
#            os.path.abspath(out_file)]
#    joblib.dump(args, out_file+'.cmd')
#    print ' '.join(args)
#    subprocess.call(args, cwd = COMPINVMEAS_OPENCL_PATH)
#    if not os.access(out_file, os.R_OK):
#        raise ValueError, 'Creation of file %s failed!'%out_file
#    out = mmap_binary_interval_vector(2, out_file)
#    base_lyap = RI(out[0][0], out[0][1]) / K
#    base_logdf_LInfEst = RI(RI(out[1][0], out[1][1]).abs().upper())
#
#    # the following code makes different attempts in estimating the
#    # lyapunov exponent, taking smaller and smaller neighborhood at the
#    # singularities.
#
#    SMALLEST_ERROR_SO_FAR = 1000000 # search for the smallest possible error
#    lyapBestRigorousEstimate = RI(0)
#
#    xi = noise_size_rel
#    measLinfinity_est = 1 / xi     # norm-infinity of the inv-measure
#
#    print 'Lyapunov, computing best intervals around the singularities'
#    for t in range(1,10):
#        for t1 in range(1,10):
#          for t3 in range(1,10):
#            lyap = base_lyap
#            logdf_LInfEst = base_logdf_LInfEst
#
#            # sing1
#            s1_boundary1 = singul1 - (RI(2)**(-1-t1))/K
#            s1_boundary2 = singul1 + (RI(2)**(-1-t1))/K
#
#            loc_lyap, loc_logdf_LInfEst = compute_component_at_s1(D, K, meas_vec,
#                                                                  measLinfinity_est, idx1,
#                                                                  s1_boundary1, s1_boundary2)
#            lyap += loc_lyap
#            logdf_LInfEst = logdf_LInfEst.max(loc_logdf_LInfEst)
#            
#            # boundaris around sing2
#            s2_boundary1 = singul2 - (singul2 - D.field(idx2) / K) * 2**(-t3)
#            s2_boundary2 = singul2 + (D.field((idx2 + 2) / K) - singul2) * 2**(-t)
#
#            loc_lyap, loc_logdf_LInfEst = compute_component_at_s2(D, K, meas_vec,
#                                                                  measLinfinity_est, idx2,
#                                                                  s2_boundary1, s2_boundary2)
#            lyap += loc_lyap
#            logdf_LInfEst = logdf_LInfEst.max(loc_logdf_LInfEst)
#            print "WW:",logdf_LInfEst.lower(), logdf_LInfEst.upper(),\
#                loc_logdf_LInfEst.lower(),loc_logdf_LInfEst.upper()
#
#
#            # interval rigorously containing the Lyapunov exponent
#            lyapErrEst = measureL1_Err * logdf_LInfEst
#            lyapRigorous = lyap + lyapErrEst.union(- lyapErrEst) 
#
#            error = lyapRigorous.upper() - lyapRigorous.lower()
#            if error < SMALLEST_ERROR_SO_FAR:
#                SMALLEST_ERROR_SO_FAR = error
#                lyapBestRigorousEstimate = lyapRigorous
#                print logdf_LInfEst.lower(),'~',logdf_LInfEst.upper()
#                print t1, t3, t
#                print s1_boundary1
#                print s1_boundary2
#                print s2_boundary1
#                print s2_boundary2
#
#    print 'Lyapunov, done'
#    return lyapBestRigorousEstimate


class ObsSingularity:
    def __init__(self, value, left_integral_estimate, right_integral_estimate):
        self.value = value
        self.left_integral_estimate = left_integral_estimate
        self.right_integral_estimate = right_integral_estimate

def estimate_observable_L1Linf(K, RI, meas_vec, meas_file,
                               meas_L1_error, meas_Linf_apriori_bound,
                               meas_Linf_estimate,
                               obs_vec, obs_file,
                               obs_singularities,
                               compute_observable,
                               approx_out_steps, approx_in_steps):

    # invoke external C++ program to compute L1 norm of meas_file
    # the measure vector does not have to be of norm 1, in case it is not
    # everything will be scaled to the correct proportion (including the error)
    out_file = 'out.dat'
    args = ['./ComputeL1Norm',
            str(K),
            os.path.abspath(meas_file),
            os.path.abspath(out_file)]
    joblib.dump(args, out_file+'.cmd')
    print ' '.join(args)
    subprocess.call(args, cwd = COMPINVMEAS_OPENCL_PATH)
    if not os.access(out_file, os.R_OK):
        raise ValueError, 'Creation of file %s failed!'%out_file

    out = mmap_binary_interval_vector(1, out_file)
    meas_L1_norm = RI(1) #RI(out[0][0], out[0][1])
    meas_Linf_bound = meas_L1_norm * meas_Linf_apriori_bound

    def meas_Linf_in_interval(intv):
        imin = Integer(floor(intv.lower() * len(meas_Linf_estimate)))
        imax = Integer(ceil(intv.upper() * len(meas_Linf_estimate)))
        return max(meas_Linf_estimate[i] for i in range(imin, imax+1))
    
    range_spec = "0-"
    big_step = Integer(round(1.4**approx_out_steps))
    for s in obs_singularities:
        v = Integer(round(s.value * K).lower()) - 1
        start = v - big_step
        end   = v + 2 + big_step
        range_spec += "%d,%d-"%(start, end)
    range_spec += "%d"%K

    # invoke external C++ program for the biggest part
    # of the computation (not close to singularities).
    out_file = 'out.dat'
    args = ['./ComputeRangeDotProduct',
            str(K),
            os.path.abspath(meas_file),
            os.path.abspath(obs_file),
            range_spec,
            os.path.abspath(out_file)]
    joblib.dump(args, out_file+'.cmd')
    print ' '.join(args)
    subprocess.call(args, cwd = COMPINVMEAS_OPENCL_PATH)
    if not os.access(out_file, os.R_OK):
        raise ValueError, 'Creation of file %s failed!'%out_file

    # these are the estimates for the result, the Linf estimate of
    # the observable, in the region we selected.
    out = mmap_binary_interval_vector(2, out_file)
    base_result = RI(out[0][0], out[0][1]) / K
    base_obs_extrema = RI(out[1][0], out[1][1])

    print "Dot_base: ",base_result.lower(), base_result.upper()

    sing_est_data = []
    for lev in range(2*len(obs_singularities)):
        i = int(lev / 2)
        left_side = is_even(lev) # taking care of left or right hand neightborhood?
        possibilities = []

        s = obs_singularities[i]
        v = Integer(round(s.value * K).lower()) - 1

        for m in range(0, approx_out_steps+1):
            step = Integer(round(1.4**m)) if m > 0 else 0
            far_outer_range = [v-big_step, v-step] if left_side \
                         else [v+2+step, v+2+big_step]
            separator = RI(v-step)/K if left_side else RI(v+2+step)/K

            obs_extrema_expansion = None if far_outer_range[0] == far_outer_range[1] \
                            else union_in_interval(obs_vec, RI, far_outer_range) 

            my_meas_Linf_bound = RI(meas_Linf_in_interval(separator.union(s.value)))
            #print 'MY BOUND:',my_meas_Linf_bound.upper(),'(abs=',meas_Linf_bound.upper(),')'
            #print 'in interval [',separator.union(s.value).lower(),',',\
            #     separator.union(s.value).upper(),']'
            #my_meas_Linf_bound = meas_Linf_bound
            result_contribute = dot_prod_in_interval(meas_vec, obs_vec,
                                                     RI, far_outer_range) \
                + my_meas_Linf_bound.union(0) * (
                    s.left_integral_estimate(separator) if left_side \
                    else s.right_integral_estimate(separator)
                )
            possibilities.append( (lev,m, result_contribute, obs_extrema_expansion) )

        far_outer_range = [v-big_step, v] if left_side \
                         else [v+2, v+2+big_step]
        
        curr_obs_extrema_expansion = None if far_outer_range[0] == far_outer_range[1] \
                                     else union_in_interval(obs_vec, RI, far_outer_range)
        curr_result_contribute = dot_prod_in_interval(meas_vec, obs_vec,
                                                     RI, far_outer_range)
        
        for m in range(1, approx_in_steps+1):
            extrema = (RI(v)/K) if left_side else (RI(v+2)/K)
            separator = s.value + (extrema-s.value)*(RI(1.4)**(-m))
            far_inner_range = [v, ceil((separator*K).upper())] if left_side \
                                else [floor((separator*K).lower()), v+2]
            far_inner_int = (RI(v)/K).union(separator) if left_side \
                                else (RI(v+2)/K).union(separator)

            far_inner_extrema = compute_observable(far_inner_int)
            obs_extrema_expansion = far_inner_extrema if not curr_obs_extrema_expansion \
                                    else curr_obs_extrema_expansion.union(far_inner_extrema)

            my_meas_Linf_bound = RI(meas_Linf_in_interval(separator.union(s.value)))
            result_contribute = curr_result_contribute + \
                    mass_in_interval(meas_vec, RI, far_inner_range).union(0) * \
                        far_inner_extrema \
                    + my_meas_Linf_bound.union(0) * (
                      s.left_integral_estimate(separator) if left_side \
                      else s.right_integral_estimate(separator)
                )
            possibilities.append( (lev,-m, result_contribute, obs_extrema_expansion) )

        sing_est_data.append(possibilities)
        
    def enum_matches(lev, part_result, part_obs_extrema, info):
        if lev >= len(sing_est_data):
            yield (part_result, part_obs_extrema, info)
            return
        for (l,m, result_contribute, obs_extrema_expansion) in sing_est_data[lev]:
            new_info = info + "_%d"%(m) + "_[%f,%f]"%(result_contribute.lower(),
                                                      result_contribute.upper())
            new_result = part_result + result_contribute
            new_obs_extrema = part_obs_extrema if not obs_extrema_expansion \
                              else obs_extrema_expansion.union(part_obs_extrema)
            for z in enum_matches(lev+1, new_result, new_obs_extrema, new_info):
                yield z

    lower_bound = RealField(prec = RI.prec())('-Infinity')
    upper_bound = RealField(prec = RI.prec())('+Infinity')
    lower_info = ""
    upper_info = ""
    
    for (result, obs_extrema, info) in enum_matches(0, base_result, base_obs_extrema, ""):

        # check out the estimate of the integration over Y contained in X
        factor = (RI(obs_extrema.upper())-RI(obs_extrema.lower()))/2 + \
                  abs(RI(obs_extrema.upper())+RI(obs_extrema.lower()))/4
        err_est = factor * meas_L1_error
        rig_result = result + err_est.union(-err_est)
        info += "_%f"%err_est.upper()

        if rig_result.lower() > lower_bound:
            lower_bound = rig_result.lower()
            lower_info = info
        if rig_result.upper() < upper_bound:
            upper_bound = rig_result.upper()
            upper_info = info

    # return result rescaling back the measure to 1 (may be necessary)
    return RI(lower_bound, upper_bound) / meas_L1_norm, lower_info, upper_info
