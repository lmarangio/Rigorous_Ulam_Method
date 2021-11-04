
from partition import equispaced
from matrix_io import save_matrix_market, save_binary_matrix, dump_while_assembling
from sparse import sage_sparse_to_scipy_sparse, sparse_matvec,\
     interval_norm1_error, max_nonzeros_per_row, norm1, interval_norminf_error
from ulam import *
from partition import equispaced
import numpy as np
import os
import joblib
import gc

import time
from matrix_io import *
from nagumosato import *
from noise_measure_computation import *
from noise_observable_estimation import *

# Computes rigorous intervals containing the derivative in some intervals.
# Of course close to the singularities these values will go to +/- infinity
def compute_log_deriv(D, K):
    if not os.path.exists(D.name):
        os.makedirs(D.name)

    fingerprint = 'P%d_K%d' % (D.field.prec(), K)
    logderiv_file = D.name+'/logderiv_'+fingerprint
    print logderiv_file
    if not os.access(logderiv_file, os.R_OK):
        print 'Computing log-deriv data...'
        RI53 = RealIntervalField(53)
        retv = VectorSpace(RI53, K)(0)
        start_time = time.time()
        for i in range(K):
            retv[i] = D.f_prime(D.field(i,(i+1))/K).abs().log()
            if ((i+1)%2014==0):
                elapsed = time.time() - start_time
                to_completion = elapsed / (i+1) * (K-i-1)
       	        print ("%d (%.02f%%, elapsed: %s, ETA: %s) " % (
                   i,
                   RealField(prec=53)((i+1)*100.0/K),
                   show_time(elapsed),
                   show_time(to_completion)
                ) )
        print 'Saving log-deriv...'       
        save_binary_interval_vector(retv, logderiv_file) 
        retv = mmap_binary_interval_vector(K, logderiv_file)
        print 'Done.'
        return retv, logderiv_file
    else: 
        print 'M-mapping log-deriv...'
        retv = mmap_binary_interval_vector(K, logderiv_file)
        print 'Done.'       
        return retv, logderiv_file

# this is a list of all singularities of our observable, with functions
# that estimate the L1 norm close to the singularity.
def create_log_deriv_singularities_info(D):
    return [
        ObsSingularity(D.sep1,
            lambda p: estimate_sing_integral(D.sep1, p, D.f_second(D.sep1.union(p))),
            lambda q: estimate_sing_integral(D.sep1, q, D.f_second(D.sep1.union(q)))),
        ObsSingularity(D.sep2,
            lambda p: estimate_sing_integral(D.sep2, p, D.f_second(D.sep2.union(p))),
            lambda q: estimate_sing_integral(D.sep2, q, D.f_second(D.sep2.union(q))))
    ]

def make_computation(D, K, Kcoarse, num_iter, coarse_noise_abs, Kest, shift=0):
    dir = D.name + '_results'
    if not os.path.exists(dir):
        os.makedirs(dir)
    fingerprint = 'P%d_K%d_C%d_N%d_I%d_E%d_S%d' % (D.field.prec(), K, Kcoarse,
                                                   coarse_noise_abs, num_iter, Kest, shift)
    riglyap_file = dir+'/riglyap_' + fingerprint
    
    if not os.access(riglyap_file, os.R_OK):

        # compute invariant measure with rigorous error
        meas_vec, meas_L1_error, meas_file, meas_Linf_estimate = \
            compute_measure(D, K, Kcoarse,
                            num_iter, coarse_noise_abs, Kest, shift=shift)

        # compute the observable
        obs_vec, obs_file = compute_log_deriv(D, K)

        obs_sing = create_log_deriv_singularities_info(D)

        meas_Linf_apriori_bound = D.field(Kcoarse) / coarse_noise_abs
        lyap_rig, linfo, uinfo = estimate_observable_L1Linf(K, D.field,
                                meas_vec, meas_file,
                                meas_L1_error,
                                meas_Linf_apriori_bound,
                                meas_Linf_estimate,
                                obs_vec, obs_file,
                                obs_sing,
                                lambda x: log(abs(D.f_prime(x))),
                                20, 7)

        joblib.dump([lyap_rig, linfo, uinfo], riglyap_file)
    else:
        print riglyap_file
        lyap_rig, linfo, uinfo = joblib.load(riglyap_file)

    print "Lyap rig: [", str(lyap_rig.lower()),",", str(lyap_rig.upper()),"]"
    print "Computed with ",linfo,' and ',uinfo
    return lyap_rig

#K = 2**25        # for computing the invariant measure
#Kcoarse = 2**20  # for contraction checking
#num_iter = 70    # try this many iterations, looking for a contraction
#Kest = 2**16     # for delta^2 improvement
#coarse_noise_abs = 112
#make_computation(D, K, Kcoarse, num_iter, coarse_noise_abs, Kest)

# make computation for many parameter values.
# ok having a list is not ideal, but will do
# parameters are:
#  * D:                the system
#  * K:                size of the fine partition, for the final invariant measure
#  * Kcoarse:          size of the coarser partition for checking contraction, via coarse-fine
#  * num_iter:         number of the iterations to try, around 60-70 gives a contraction here
#  * coarse_noise_abs: size of noise, with respect to coarser size (size is coarse_noise_abs/Kcoarse)
#  * Kest:             size of the coarser partition used for verification in the delta^2

D = NagumoSato(256)
params = [
    (D, 2**25, 2**16, 60, 690, 2**14), #0.01052
     (D, 2**25, 2**16, 60, 630, 2**14),
     (D, 2**25, 2**16, 60, 578, 2**14),
    (D, 2**25, 2**16, 60, 530, 2**14), #0.008087
     (D, 2**25, 2**16, 60, 486, 2**14),
     (D, 2**25, 2**16, 60, 446, 2**14),
    (D, 2**25, 2**16, 60, 408, 2**14), #0.006225
     (D, 2**25, 2**16, 60, 374, 2**14),
     (D, 2**25, 2**16, 60, 342, 2**14),
    (D, 2**25, 2**16, 60, 314, 2**14), #0.004791
     (D, 2**25, 2**16, 60, 288, 2**14),
     (D, 2**25, 2**16, 60, 264, 2**14),
    (D, 2**25, 2**16, 60, 242, 2**14), #0.003692
     (D, 2**25, 2**16, 60, 222, 2**14),
     (D, 2**25, 2**16, 60, 202, 2**14),
    (D, 2**25, 2**16, 60, 186, 2**14), #0.002838
     (D, 2**25, 2**16, 60, 170, 2**14),
     (D, 2**25, 2**16, 60, 154, 2**14),
    (D, 2**25, 2**16, 60, 142, 2**14), #0.002166
     (D, 2**25, 2**16, 60, 130, 2**14),
     (D, 2**25, 2**16, 60, 120, 2**14),
    (D, 2**25, 2**16, 60, 110, 2**14), #0.001678
     (D, 2**25, 2**17, 60, 200, 2**14),
     (D, 2**25, 2**17, 60, 184, 2**14),
    (D, 2**25, 2**17, 60, 168, 2**14), #0.001281
     (D, 2**25, 2**17, 60, 154, 2**14),
     (D, 2**25, 2**17, 60, 142, 2**14),
    (D, 2**25, 2**17, 60, 130, 2**14), #0.0009918
     (D, 2**26, 2**17, 60, 120, 2**14),
     (D, 2**26, 2**17, 60, 110, 2**14),
    (D, 2**26, 2**17, 60, 100, 2**14), #0.0007629
     (D, 2**26, 2**18, 60, 184, 2**15),
     (D, 2**26, 2**18, 60, 168, 2**15),
    (D, 2**26, 2**18, 60, 154, 2**15), #0.0005874
     (D, 2**26, 2**18, 60, 140, 2**15),
     (D, 2**26, 2**18, 60, 128, 2**15),
    (D, 2**26, 2**18, 60, 118, 2**15), #0.0004501
     (D, 2**26, 2**19, 65, 220,  2**16),
     (D, 2**26, 2**19, 65, 200,  2**16),
    (D, 2**26, 2**19, 65, 184,  2**16),  #0.0003509
     (D, 2**26, 2**19, 65, 166,  2**16),
     (D, 2**26, 2**19, 65, 152,  2**16),
    (D, 2**26, 2**19, 65, 140,  2**16),  #0.0002670
     (D, 2**26, 2**19, 65, 128,  2**16),
     (D, 2**26, 2**19, 65, 118,  2**16),
    (D, 2**26, 2**19, 65, 108,  2**16),  #0.0002060
     (D, 2**26, 2**20, 70, 198,  2**17),
     (D, 2**26, 2**20, 70, 182,  2**17),
    (D, 2**26, 2**20, 70, 166,  2**17),  #0.0001583
]

params = [
 #   (D, 2**24, 2**13, 60, 690, 2**13), #0.08422
 #    (D, 2**24, 2**13, 60, 630, 2**13),
 #    (D, 2**24, 2**13, 60, 578, 2**13),
 #    (D, 2**24, 2**13, 60, 530, 2**13),
 #   (D, 2**24, 2**13, 60, 486, 2**13), #0.05932
 #    (D, 2**24, 2**13, 60, 446, 2**13),
 #    (D, 2**24, 2**13, 60, 408, 2**13), 
 #    (D, 2**24, 2**14, 60, 756, 2**13),
 #   (D, 2**24, 2**14, 60, 690, 2**13), #0.04211
 #    (D, 2**24, 2**14, 60, 630, 2**13),
 #    (D, 2**24, 2**14, 60, 578, 2**13),
    (D, 2**24, 2**14, 60, 530, 2**13), #0.03234
     (D, 2**24, 2**14, 60, 486, 2**13),
     (D, 2**24, 2**14, 60, 446, 2**13),
     (D, 2**24, 2**14, 60, 408, 2**13), 
    (D, 2**24, 2**15, 60, 756, 2**13), #0.02307
     (D, 2**24, 2**15, 60, 690, 2**13),
     (D, 2**24, 2**15, 60, 630, 2**13),
    (D, 2**24, 2**15, 60, 578, 2**13), #0.01763
     (D, 2**24, 2**15, 60, 530, 2**13), 
     (D, 2**24, 2**15, 60, 486, 2**13),
    (D, 2**24, 2**15, 60, 446, 2**13), #0.01361
     (D, 2**24, 2**15, 60, 408, 2**13), 
     (D, 2**24, 2**15, 60, 374, 2**13),
    (D, 2**24, 2**16, 60, 690, 2**14), #0.01052
     (D, 2**24, 2**16, 60, 630, 2**14),
     (D, 2**24, 2**16, 60, 578, 2**14),
    (D, 2**24, 2**16, 60, 530, 2**14), #0.008087
     (D, 2**24, 2**16, 60, 486, 2**14),
     (D, 2**24, 2**16, 60, 446, 2**14),
    (D, 2**24, 2**16, 60, 408, 2**14), #0.006225
     (D, 2**24, 2**16, 60, 374, 2**14),
     (D, 2**24, 2**16, 60, 342, 2**14),
    (D, 2**24, 2**16, 60, 314, 2**14), #0.004791
     (D, 2**24, 2**16, 60, 288, 2**14),
     (D, 2**24, 2**16, 60, 264, 2**14),
    (D, 2**24, 2**16, 60, 242, 2**14), #0.003692
     (D, 2**24, 2**16, 60, 222, 2**14),
     (D, 2**24, 2**16, 60, 202, 2**14),
    (D, 2**24, 2**16, 60, 186, 2**14), #0.002838
     (D, 2**24, 2**16, 60, 170, 2**14),
     (D, 2**24, 2**16, 60, 154, 2**14),
    (D, 2**24, 2**16, 60, 142, 2**14), #0.002166
     (D, 2**24, 2**16, 60, 170, 2**14),
     (D, 2**24, 2**16, 60, 154, 2**14),
    (D, 2**24, 2**16, 60, 142, 2**14), #0.002166
     (D, 2**24, 2**16, 60, 130, 2**14),
     (D, 2**24, 2**16, 60, 120, 2**14),
    (D, 2**24, 2**16, 60, 110, 2**14), #0.001678
     (D, 2**24, 2**17, 80, 200, 2**14),
     (D, 2**24, 2**17, 80, 184, 2**14),
    (D, 2**24, 2**17, 80, 168, 2**14), #0.001281
     (D, 2**24, 2**17, 80, 154, 2**14),
     (D, 2**24, 2**17, 80, 142, 2**14),
    (D, 2**24, 2**17, 80, 130, 2**14), #0.0009918
     (D, 2**24, 2**17, 80, 120, 2**14),
     (D, 2**24, 2**17, 80, 110, 2**14),
    (D, 2**24, 2**17, 80, 100, 2**14), #0.0007629
]


for p in params:
    make_computation(*p, shift=p[4]/2)
    gc.collect()
