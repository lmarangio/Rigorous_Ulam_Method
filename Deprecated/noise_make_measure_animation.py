
from sage.all import *
import numpy as np
import os
import joblib
import subprocess
from partition import step_function, equispaced
from matrix_io import *
from noise_gpu_settings import *

def create_animation(D, params, shift = 0, mod1 = False, **plot_args):

    red_size = 2000
    plots = []

    for K, Kcoarse, num_iter, coarse_noise_abs, Kest in params:
        #fingerprint = 'P%d_K%d_C%d_N%d_I%d_E%d' % (D.field.prec(), K, Kcoarse,
        #                                       coarse_noise_abs, num_iter, Kest)
        meas_fingerprint = 'P%d_K%d_N%d_S%d%s' % (D.field.prec(), K,
                                                  coarse_noise_abs*K/Kcoarse,
                                                  shift,
                                                  '_m1' if mod1 else '')
        measure_file = D.name + '_results/measure_' + meas_fingerprint
        measure_sample_file = D.name + '_results/measure_' + \
                              meas_fingerprint + ('_red%d'%red_size)

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

            scale = 1.0 if not hasattr(D,'_scale') else RR(D._scale.center())
            noise = RR(coarse_noise_abs)/Kcoarse*scale
            plots.append( (noise, measure_sample_file) )
        else:
            print('Not found: "{0}"'.format(measure_file))
                    
    plots.sort()

    try:
        os.mkdir(D.name + '_images')
    except:
        pass

    for i in range(len(plots)):
        noise, sample = plots[i]
        vec = mmap_binary_vector(red_size, sample)
        func = step_function(vec/red_size, equispaced(red_size))
        plot = plot_step_function(func, legend_label = str(noise),
                                  **plot_args)
        plot.save_image(D.name + '_images/plot_%03d.png' % i)
