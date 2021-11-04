from partition import equispaced
from matrix_io import *
from ulam import *
from partition import equispaced
from generic_assembler import *
import subprocess
import numpy as np
import os
import joblib
import time
from matrix_io import *

from bzmodel import *
from noise_matrix_creation import *
from noise_columnwise_assembler import *
from noise_gpu_settings import *


# computes the step error in an application of the matrix P (no noise)
def compute_basic_step_error(K, abs_diam, max_nnz, norm1_error):
    Rup = RealField(rnd='RNDU')
    Rdown = RealField(rnd='RNDD')

    machineEpsilon = Rup(np.finfo(float).eps)

    gammak = machineEpsilon*max_nnz / (1-machineEpsilon*max_nnz) + machineEpsilon
    steperror = Rup(norm1_error) + Rup(gammak)

    return steperror


# compute the step error because of the application of noise,
# it depends on the number of nonzeros of the noise matrix,
# therefore depends in the resolution level too.
def compute_noise_step_error(noise_size_nnz):
    Rup = RealField(rnd='RNDU')
    Rdown = RealField(rnd='RNDD')
    machineEpsilon = Rup(np.finfo(float).eps)
    gamma_noise_k = machineEpsilon*noise_size_nnz / (1 - machineEpsilon*noise_size_nnz)
    return Rup(gamma_noise_k)


# creates the matrix, avoids recreating it if it is already saved somewhere
def create_contraction_matrix(D, K, prec):
    dir = D.name
    if not os.path.exists(dir):
        os.makedirs(dir)

    filename = dir+'/mat_%d_%d' % (K, prec)
    verif_filename = dir+'/verif_%d_%d' % (K, prec)
    data_filename = filename + '_data'

    if os.access(filename, os.R_OK):
        basic_step_error, abs_diam, max_nnz, norm1_error = joblib.load(data_filename)
        return filename, verif_filename, basic_step_error
    else:
        print '* assembling matrix'
        epsilon = D.field(2) ** (20-prec)

        P, abs_diam, max_nnz, norm1_error = columnwise_assemble(D,
                                    UlamL1(equispaced(K)),
                                    epsilon,
                                    scipy_matrix = True,
                                    verification_basis = verif_filename)
        save_scipy_to_binary_matrix(P, filename, use_double = True)

        basic_step_error = compute_basic_step_error(K, abs_diam,
                                                    max_nnz, norm1_error)
        joblib.dump([basic_step_error, abs_diam, max_nnz, norm1_error], data_filename)

        return filename, verif_filename, basic_step_error


DISABLE_REFINED_COARSE_FINE = True

def prove_contraction(D, K, noise_abs, num_iter, shift=0, mod1=False):
    # for a given K and a given noise...
    ulam_basis = UlamL1(equispaced(K))

    # the matrix will not be recreated, if it is already there
    if DISABLE_REFINED_COARSE_FINE:
        matrix_file, basic_step_error = \
                create_matrix(D, K, D.field.prec())
    else:
        matrix_file, verif_file, basic_step_error = \
                create_contraction_matrix(D, K, D.field.prec())
    step_error = (basic_step_error + compute_noise_step_error(noise_abs+1))

    # simultaneous vectors for verification
    #decrease this, if there is not enough GPU memory
    sym_vecs = 32 if K >= 2**20 else 64

    # for identifying already computed data
    fingerprint = 'P%d_K%d_N%d_I%d_S%d%s' % (D.field.prec(), K, noise_abs, num_iter, shift, '_m1' if mod1 else '')

    # create result dir if necessary
    dir = D.name+'_results'
    if not os.path.exists(dir):
        os.makedirs(dir)

    # verify contraction
    contr_file = dir+'/contr_'+fingerprint
    print contr_file
    if not os.access(contr_file, os.R_OK):
        args = ['./ContractionSpM',
            '-b', os.path.abspath(matrix_file),
            '-r', str(noise_abs),
            '-N', 'Norm_L1',
            '-F', str(num_iter),
            '-s', str(step_error),
            '-f', 'd',
            '-noise-shift', str(floor(shift)),
            '-V', str(sym_vecs),
            '-B', 'Basis_0_Np1',
            '-allow-quick-contraction-check',
            '-power-norms-file', os.path.abspath(contr_file)
        ] + COMPINVMEAS_OPENCL_FLAGS
        if mod1:
            args.append('-noise-mod-1')
        joblib.dump(args, contr_file+'.cmd')
        print ' '.join(args)
        subprocess.call(args, cwd = COMPINVMEAS_OPENCL_PATH)
        if not os.access(contr_file, os.R_OK):
            raise ValueError, 'Creation of file %s failed!'%contr_file

    if DISABLE_REFINED_COARSE_FINE:
        # compute best constant of the form sum[Ci]/(1-Cn) for some n
        Rup = RealField(rnd='RNDU', prec=53)
    
        # we call it Cip1 because the ith component is the (i+1)th in the paper,
        # that is, the vector starts with C_1
        Cip1 = [min(x, Rup(1)) for x in read_binary_vector(Rup, num_iter, contr_file, 'd')]

        Rup = RealField(rnd='RNDU', prec=53)
        FineCip1 = range(num_iter)
        for i in range(num_iter):
            # noise_abs is coarse_noise_abs, that is xi/delta_coarse
            dist_L_Lcoarse = (Rup(1)+Rup(2)+sum(2*Cip1[k] for k in range(i-1))) / noise_abs #1+2C_0+...+2C_i
            FineCip1[i] = min(Rup(1), (Cip1[i-1] if i>0 else Rup(1)) + dist_L_Lcoarse)

        Consts = range(num_iter)
        for i in range(num_iter):
            SumFineCip1 = Rup(1) + sum(FineCip1[j] for j in range(i))
            # if stopping at this i, we compute sum_Cj/(1-Ci), and set the other values
            Consts[i] = (SumFineCip1/(1-FineCip1[i]), SumFineCip1, FineCip1[i], i+1,
                         (Cip1[i-1] if i>0 else Rup(1))) # index is i+1 actually

        return min(Consts)

        
    # compute contraction of the "verif" basis
    contr_verif_file = dir+'/verif_'+fingerprint
    print contr_verif_file
    if not os.access(contr_verif_file, os.R_OK):
        args = ['./ContractionSpM',
            '-b', os.path.abspath(matrix_file),
            '-r', str(noise_abs),
            '-N', 'Norm_L1',
            '-F', str(num_iter),
            '-s', str(step_error),
            '-f', 'd',
            '-V', str(sym_vecs),
            '-noise-shift', str(floor(shift)),
            '-stop-at-norm', '0.08', # should be good enough
            '-start-with-only-noise',
            '-basis-file-path', os.path.abspath(verif_file),
            '-basis-norms-collection-file', os.path.abspath(contr_verif_file)
        ] + COMPINVMEAS_OPENCL_FLAGS
        if mod1:
            args.append('-noise-mod-1')
        joblib.dump(args, contr_verif_file+'.cmd')
        subprocess.call(args, cwd = COMPINVMEAS_OPENCL_PATH)
        if not os.access(contr_verif_file, os.R_OK):
            raise ValueError, 'Creation of file %s failed!'%contr_verif_file

    # compute contraction of the "noise-gaps" basis
    contr_noisgaps_file = dir+'/gaps_'+fingerprint
    print contr_noisgaps_file
    if not os.access(contr_noisgaps_file, os.R_OK):
        args = ['./ContractionSpM',
            '-b', os.path.abspath(matrix_file),
            '-r', str(noise_abs),
            '-N', 'Norm_L1',
            '-F', str(num_iter),
            '-s', str(step_error),
            '-f', 'd',
            '-V', str(sym_vecs),
            '-noise-shift', str(floor(shift)),
            '-stop-at-norm', '0.08', # should be good enough
            '-B', 'Basis_Noise_Gap',
            '-basis-norms-collection-file', os.path.abspath(contr_noisgaps_file)
        ] + COMPINVMEAS_OPENCL_FLAGS
        if mod1:
            args.append('-noise-mod-1')
        joblib.dump(args, contr_noisgaps_file+'.cmd')
        print ' '.join(args)
        subprocess.call(args, cwd = COMPINVMEAS_OPENCL_PATH)
        if not os.access(contr_noisgaps_file, os.R_OK):
            raise ValueError, 'Creation of file %s failed!'%contr_noisgaps_file

    # combine the contractions of the above estimate
    compound_norms_file = dir+'/compound_norms_'+fingerprint
    print compound_norms_file
    if not os.access(compound_norms_file, os.R_OK):
        args = ['./ComputeNormsFromLocal',
                str(K),
                str(num_iter),
                str(noise_abs),
                os.path.abspath(contr_noisgaps_file),
                os.path.abspath(contr_verif_file),
                os.path.abspath(compound_norms_file)
        ]
        if mod1:
            args.append('-noise-mod-1')
        joblib.dump(args, compound_norms_file+'.cmd')
        print ' '.join(args)
        subprocess.call(args, cwd = COMPINVMEAS_OPENCL_PATH)
        if not os.access(compound_norms_file, os.R_OK):
            raise ValueError, 'Creation of file %s failed!'%compound_norms_file

    # compute best constant of the form sum[Ci]/(1-Cn) for some n
    Rup = RealField(rnd='RNDU', prec=53)
    compound_norms = read_binary_vector(Rup, num_iter, compound_norms_file, 'd')

    # we call it Cip1 because the ith component is the (i+1)th in the paper,
    # that is, the vector starts with C_1
    Cip1 = [min(x, Rup(1)) for x in read_binary_vector(Rup, num_iter, contr_file, 'd')]

    Rup = RealField(rnd='RNDU', prec=53)
    FineCip1 = range(num_iter)
    for i in range(num_iter):
        dist_L_Lcoarse = (Rup(1)+sum(compound_norms[j] for j in range(i+1))) / noise_abs
        FineCip1[i] = min(Rup(1), (Cip1[i-1] if i > 0 else Rup(1)) + dist_L_Lcoarse)

    Consts = range(num_iter)
    for i in range(num_iter):
        SumFineCip1 = Rup(1) + sum(FineCip1[j] for j in range(i))
        Consts[i] = (SumFineCip1/(1-FineCip1[i]), SumFineCip1, FineCip1[i], i+1,
                     (Cip1[i-1] if i>0 else Rup(1))) # index is i+1 actually

    return min(Consts)
