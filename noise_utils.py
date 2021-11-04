
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
from noise_measure_computation import *
from noise_observable_estimation import *
from plotting import *
from noise_gpu_settings import *

# helper function, for plotting measure
def create_plot(measure_file, noise, red_size, K):
    measure_sample_file = measure_file + ('_red%d'%red_size)
    
    if os.access(measure_file, os.R_OK):
        if not os.access(measure_sample_file, os.R_OK):
            args = ['./ComputeReduction',
                    str(K),
                    os.path.abspath(measure_file),
                    str(red_size),
                    os.path.abspath(measure_sample_file)
            ]
            print ' '.join(args)
            subprocess.call(args, cwd = COMPINVMEAS_OPENCL_PATH)
            if not os.access(measure_sample_file, os.R_OK):
                raise ValueError, 'Creation of file %s failed!'%measure_sample_file

        vec = mmap_binary_vector(red_size, measure_sample_file)
        func = step_function(vec/red_size, equispaced(red_size))
        plot = plot_step_function(func, legend_label = str(noise),
                          # uncomment if you want y in logarithmic scale:
                          # scale='semilogy',
                          ymax=5, ymin = 0.01)
        plot.show()



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
            retv[i] = D.f_prime_log_abs(D.field(i,(i+1))/K)
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
        print 'Done.'
        return retv, logderiv_file
    else: 
        print 'M-mapping log-deriv...'
        retv = mmap_binary_interval_vector(K, logderiv_file)
        print 'Done.'       
        return retv, logderiv_file

def compute_rot_num(D, K):
    if not os.path.exists(D.name):
        os.makedirs(D.name)

    fingerprint = 'P%d_K%d' % (D.field.prec(), K)
    rotnum_file = D.name+'/rotnum_'+fingerprint
    print rotnum_file
    if not os.access(rotnum_file, os.R_OK):
        print 'Computing rot-num data...'
        RI53 = RealIntervalField(53)
        retv = VectorSpace(RI53, K)(0)
        start_time = time.time()
        for i in range(K):
            retv[i] = D.f_lift(D.field(i,(i+1))/K)
            if ((i+1)%2014==0):
                elapsed = time.time() - start_time
                to_completion = elapsed / (i+1) * (K-i-1)
                print ("%d (%.02f%%, elapsed: %s, ETA: %s) " % (
                   i,
                   RealField(prec=53)((i+1)*100.0/K),
                   show_time(elapsed),
                   show_time(to_completion)
                ) )
        print 'Saving rot-num...'       
        save_binary_interval_vector(retv, rotnum_file) 
        print 'Done.'
        return retv, rotnum_file
    else: 
        print 'M-mapping rot-num...'
        retv = mmap_binary_interval_vector(K, rotnum_file)
        print 'Done.'       
        return retv, rotnum_file