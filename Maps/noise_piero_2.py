
from noise_utils import *
from piero import *
import joblib
        
# Create dynamical system
D = PieroModel(4098)

# Step 1: plot the dynamical system
#plot_dynamic(D).show()

# Parameters for Step 2.
# in this example, set the noise to 164/2^14 ~= 1/100, and
# we expect the system to contract in about 60 steps
# that is that L^60 is small enough (<< 1)

#K = 2**18
#Kcoarse = 2**13
#num_iter = 60
#coarse_noise_abs = 82
#Kest = 2**11

params_list = [
    [2**14, 2**14, 500, 1200, 2**12],
    [2**14, 2**14, 500, 1000, 2**12],
    [2**14, 2**14, 500, 900, 2**12],
    [2**14, 2**14, 500, 800, 2**12],
    [2**14, 2**14, 500, 700, 2**12],
    [2**14, 2**14, 500, 600, 2**12],
    [2**14, 2**14, 500, 500, 2**12],
    [2**14, 2**14, 600, 450, 2**12],
    [2**14, 2**14, 600, 400, 2**12],
    [2**14, 2**14, 600, 350, 2**12],
    [2**14, 2**14, 600, 300, 2**12],
    [2**14, 2**14, 600, 250, 2**12],
    [2**14, 2**14, 600, 225, 2**12],
    [2**14, 2**14, 600, 200, 2**12],
    [2**14, 2**14, 600, 175, 2**12],
    [2**14, 2**14, 600, 150, 2**12],
    [2**14, 2**14, 600, 125, 2**12],
    [2**14, 2**14, 600, 100, 2**12],
]

shift=0
mod1=False
    
for K, Kcoarse, num_iter, coarse_noise_abs, Kest in params_list:

    fine_noise_abs = coarse_noise_abs * K / Kcoarse
    fine_shift = shift * K / Kcoarse
    noise_size_rel = RR(coarse_noise_abs) / Kcoarse

    # create matrix, or just load relevant info
    matrix_file, basic_step_error = create_matrix(D, K, D.field.prec())

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

    # Step 3: plot the invariant measure
    #create_plot(meas_vec_file, RR(coarse_noise_abs)/Kcoarse, 10000, K)

from noise_make_measure_animation import *
create_animation(D, params_list, mod1=mod1, ymax = 5, ymin = 0)

from noise_make_graph import *
create_graph_and_sheet(D, params_list, mod1=mod1)
