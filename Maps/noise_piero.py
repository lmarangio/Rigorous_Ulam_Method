
from noise_utils import *
from piero import *
import joblib
        
# Create dynamical system
D = PieroModel(4098)

# Step 1: plot the dynamical system
plot_dynamic(D).show()

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
    [2**17, 2**14, 500, 1200, 2**12],
    [2**17, 2**14, 500, 1000, 2**12],
    [2**17, 2**14, 500, 800, 2**12],
    [2**17, 2**14, 500, 700, 2**12],
    [2**17, 2**14, 500, 600, 2**12],
    [2**17, 2**14, 500, 500, 2**12],
    [2**17, 2**14, 600, 450, 2**12],
    [2**17, 2**14, 600, 400, 2**12],
    [2**17, 2**14, 600, 350, 2**12],
]

shift=0
mod1=False
    
for K, Kcoarse, num_iter, coarse_noise_abs, Kest in params_list:

    # Step 2: compute the invariant measure
    meas_vec, meas_L1_error, meas_file, meas_Linf_estimate = \
        compute_measure(D, K, Kcoarse,
                        num_iter, coarse_noise_abs, Kest, shift=shift, mod1=mod1)


    # Step 3: plot the invariant measure
    create_plot(meas_file, RR(coarse_noise_abs)/Kcoarse, 10000, K)

    # Step 4: compute the Lyapunov exponent
    # compute the observable (log|f'| in this case)
    obs_vec, obs_file = compute_log_deriv(D, K)

    # we currently assume that there are no singularities...
    obs_sing = [] # create_log_deriv_singularities_info(D)

    # estimate the observable
    meas_Linf_apriori_bound = D.field(Kcoarse) / coarse_noise_abs
    lyap_rig, linfo, uinfo = estimate_observable_L1Linf(K, D.field,
                                    meas_vec, meas_file,
                                    meas_L1_error,
                                    meas_Linf_apriori_bound,
                                    meas_Linf_estimate,
                                    obs_vec, obs_file,
                                    obs_sing,
                                    lambda x: log(abs(D.f_prime(x))), 20, 7)

    # print the rigorous lyapunov exponent
    print "Lyap rig: [", str(lyap_rig.lower()),",", str(lyap_rig.upper()),"]"

    fingerprint = 'P%d_K%d_C%d_N%d_I%d_E%d_S%d%s' % (D.field.prec(), K, Kcoarse,
                                                     coarse_noise_abs, num_iter, Kest, shift,
                                                     '_m1' if mod1 else '')
    riglyap_file = D.name+'_results/riglyap_' + fingerprint
    joblib.dump([lyap_rig, linfo, uinfo], riglyap_file)

#from noise_make_measure_animation import *
#create_animation(D, params_list)

from noise_make_graph import *
create_graph_and_sheet(D, params_list)
