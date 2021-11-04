
from partition import equispaced
from matrix_io import save_matrix_market, save_binary_matrix, dump_while_assembling
from sparse import sage_sparse_to_scipy_sparse, sparse_matvec,\
     interval_norm1_error, max_nonzeros_per_row, norm1, interval_norminf_error
from ulam import *
from partition import equispaced
import numpy as np
import os
import joblib

from matrix_io import *
from bzmodel import *
from noise_specialized_measure_error import *
from noise_contraction_proof import *
from noise_matrix_creation import *
from noise_gpu_settings import *

def compute_noise_step_error(noise_size_nnz):
    Rup = RealField(rnd='RNDU')
    Rdown = RealField(rnd='RNDD')
    machineEpsilon = Rup(np.finfo(float).eps)
    gamma_noise_k = machineEpsilon*noise_size_nnz / (1 - machineEpsilon*noise_size_nnz)
    return Rup(gamma_noise_k)

def compute_log_deriv(D, K):
    RI53 = RealIntervalField(53)
    return [RI53(D.f_prime(D.field(i,(i+1))/K).abs().log()) for i in range(K)]


def compute_measure(D, K, Kcoarse, num_iter, coarse_noise_abs, Kest, shift=0, mod1=False):
    fine_noise_abs = coarse_noise_abs * K / Kcoarse
    fine_shift = shift * K / Kcoarse
    noise_size_rel = RR(coarse_noise_abs) / Kcoarse

    # prove contraction, at coarse level
    coarse_noise_abs = Kcoarse*noise_size_rel
    estFactor, sumCi, alpha, N, alpha_contr = prove_contraction(D, Kcoarse,
                                                                coarse_noise_abs, num_iter, shift, mod1)
    print 'Contraction proved: sumCi = ', sumCi, ', alpha = ', alpha, ', N = ', N, ',mixing =', (1-(alpha + (100/K)*(2*sumCi+1)))/N

    # create matrix, or just load relevant info
    matrix_file, basic_step_error = create_matrix(D, K, D.field.prec())

    aprioriErr = estimate_L1_error_apriori(K, noise_size_rel, alpha, sumCi)
    print 'a priori error: ', aprioriErr, ' (without numeric error)'

    fine_step_error = basic_step_error + \
                  compute_noise_step_error(fine_noise_abs + 1)
    residue = fine_step_error*2

    # create result dir if necessary
    dir = D.name+'_results'
    if not os.path.exists(dir):
        os.makedirs(dir)

    # compute the invariant measure, by interation
    fingerprint = 'P%d_K%d_N%d_S%d%s' % (D.field.prec(), K, fine_noise_abs,shift, '_m1' if mod1 else '')
    meas_vec_file = dir+'/measure_'+fingerprint
    print meas_vec_file
    if not os.access(meas_vec_file, os.R_OK):
        args = ['./IterateSpM',
            '-b', os.path.abspath(matrix_file),
            '-f', 'd',
            '-c', # on the CPU at the moment, sob...
            '-r', str(floor(fine_noise_abs)),
            '-noise-shift', str(floor(fine_shift)),
            '-n', '8000',
            '-norm-type', 'Norm_L1',
            '-step-error', str(fine_step_error),
            '-relative-residue', str(residue),
            '-o', os.path.abspath(meas_vec_file)]
        if mod1:
            args.append('-noise-mod-1')
        joblib.dump(args, meas_vec_file+'.cmd')
        print ' '.join(args)
        subprocess.call(args, cwd = COMPINVMEAS_OPENCL_PATH)
        if not os.access(meas_vec_file, os.R_OK):
            raise ValueError, 'Creation of file %s failed!'%meas_vec_file

    # let w be the difference between the computed vector and the eigenvector. Then:
    #   |L^N w - w| < NR
    #   |L^N w - w| > (1-alpha)|w|
    # therefore
    #   |w| < NR(1-alpha)
    numeric_error = N * residue * (1-alpha)
    print 'Numeric error: ', numeric_error

    print 'M-mapping measure...'
    meas_vec = mmap_binary_vector(K, meas_vec_file, fmt = 'd')
    if any(np.isnan(x) for x in meas_vec):
        raise ValueError, "Nan values in the measure?"
    print 'Done.'

    # do our magic umma-umma to improve the error estimate
    est_fingerprint = 'P%d_K%d_C%d_N%d_I%d_E%d_S%d%s' % (D.field.prec(), K, Kcoarse,
                                                       coarse_noise_abs, num_iter,
                                                       Kest, shift, '_m1' if mod1 else '')
    rigerr_file = dir+'/rigerr_'+est_fingerprint
    print rigerr_file
    if not os.access(rigerr_file, os.R_OK):
        est_basis = UlamL1(equispaced(Kest)) #somewhat coarser basis
        prep_fingerprint = 'P%d_K%d' % (D.field.prec(), Kest)

        # auxiliary data for estimating the error, 1/2
        prep1_file = dir+'/prep1_'+prep_fingerprint
        print prep1_file
        if not os.access(prep1_file, os.R_OK):
            print 'Computing data prep1...'
            prep1 = preparation_for_estimate1(D, est_basis, 2**(30-D.field.prec()))
            joblib.dump(prep1, prep1_file)
        else:
            print 'Loading...'
            prep1 = joblib.load(prep1_file)
            anynan = lambda x: any(x[i].is_NaN() for i in range(4))
            if any(any(anynan(preim) for preim in preims) for preims in prep1):
                raise ValueError, "Nan values in prep1?"
            print 'Done.'

        # auxiliary data for estimating the error, 2/2
        prep2_file = dir+'/prep2_'+prep_fingerprint
        print prep2_file
        if not os.access(prep2_file, os.R_OK):
            print 'Computing data prep2...'
            prep2 = preparation_for_estimate2(D, est_basis)
            joblib.dump(prep2, prep2_file)
        else:
            print 'Loading...'
            prep2 = joblib.load(prep2_file)
            if any(x.is_NaN() for x in prep2):
                raise ValueError, "Nan values in prep2?"
            print 'Done.'

        print 'Estimating...'
        L1error, apriori_L1error, Linf_estimate = \
            estimate_L1_error_aposteriori(prep1, prep2,
                                          meas_vec, numeric_error,
                                          UlamL1(equispaced(K)), noise_size_rel,
                                          alpha, sumCi,
                                          shift, mod1)
        joblib.dump([L1error, apriori_L1error, Linf_estimate], rigerr_file)
    else:
        L1error, apriori_L1error, Linf_estimate = joblib.load(rigerr_file)

    print 'a posteriori:',L1error,'; a priori:', apriori_L1error

    return meas_vec, L1error, meas_vec_file, Linf_estimate
