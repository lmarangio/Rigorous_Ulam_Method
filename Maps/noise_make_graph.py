
from sage.all import *
import math
import numpy as np
import os
import joblib
from bzmodel import *
from noise_contraction_proof import *

def expnot(num, dig = 3):
    if num.is_NaN():
        return 'NaN'
    e = 0
    while 10**e * num < 0.1:
        e += 1
    retv = ""
    if e > 0:
        num *= 10**e
    fmt = "%%.%df" % dig
    retv += fmt % num
    if e > 0:
        retv += "\\times 10^{-%d} " % e
    return retv


def expnotint(intv, dig = 3, sep = 3):
    if intv.is_NaN():
        return 'NaN'
    e = 0
    while ((10**e) * intv).abs().upper() < 1:
        e += 1
    retv = ""
    if e > 0:
        intv *= 10**e
    if False and intv.lower() < 0 and intv.upper() >= 0:
        fmt = "%%.%df" % dig
        retv += '['+(fmt%intv.lower())+','+(fmt%intv.upper())+']'
        if e > 0:
            retv += "\\times 10^{-%d} " % e
        return retv
    l = str(intv.lower())
    u = str(intv.upper())
    i = 0
    while i < len(l) and i < len(u):
        if l[i] != u[i]:
            break
        retv += l[i]
        i+=1
    try:
        usep = max(i+sep,u.index('.')+sep+1)
    except:
        usep = i+sep
    try:
        lsep = max(i+sep, l.index('.')+sep+1)
    except:
        lsep = i+sep
    retv += "^{"+u[i:(usep)]+"}"
    retv += "_{"+l[i:(lsep)]+"}"
    if e > 0:
        retv += "\\times 10^{-%d} " % e
    return retv

def create_graph_and_sheet(D, params, shift = 0, mod1 = False, **plot_args):
    myplot = plot([])

    data_file = file('data_sheet.tex', 'w')
    header = ['delta','delta_contr','delta_est','noise','n_contr','alpha_contr','alpha',
                      'sumCi', 'l1apriori', 'l1err', 'lyap']
    data_file.write("\t".join(header)+"\n")

    print([shift, mod1])
    for K, Kcoarse, num_iter, coarse_noise_abs, Kest in params[:-1]:
        fingerprint = 'P%d_K%d_C%d_N%d_I%d_E%d_S%d%s' % (D.field.prec(), K, Kcoarse,
                                                         coarse_noise_abs, num_iter, Kest,
                                                         shift,
                                                         '_m1' if mod1 else '')
        rigerr_file = D.name+'_results/rigerr_' + fingerprint
        riglyap_file = D.name+'_results/riglyap_' + fingerprint

        if os.access(riglyap_file, os.R_OK):
            estFactor, sumCi, alpha, N, alpha_contr = prove_contraction(D, Kcoarse,
                                                                        coarse_noise_abs, num_iter,
                                                                        shift = shift, mod1 = mod1)

            rigerr = joblib.load(rigerr_file)
            lyap_rig,li,ui = joblib.load(riglyap_file)
            scale = 1.0 if not hasattr(D,'_scale') else RR(D._scale.center())
            noise = RR(coarse_noise_abs)/Kcoarse*scale

            data = []
            Klog = Integer(RR(Integer(K).log(2)).round())
            Kcoarselog = Integer(RR(Integer(Kcoarse).log(2)).round())
            Kestlog = Integer(RR(Integer(Kest).log(2)).round())
            data.append("2^{-"+str(Klog)+"}")
            data.append("2^{-"+str(Kcoarselog)+"}")
            data.append("2^{-"+str(Kestlog)+"}")
            #data.append(str(coarse_noise_abs))
            data.append(expnot(noise))
            data.append(str(N))
            data.append("%.02g"%alpha_contr)
            data.append("%.02g"%alpha)
            data.append("%.02f"%sumCi)
            data.append(expnot(rigerr[1]))
            data.append(expnot(rigerr[0]))
            data.append(expnotint(lyap_rig))
            data_file.write(" & ".join(['$'+d+'$' for d in data])+"\\\\ \n")
            print('noise={0} -> lyap={1}'.format(noise, lyap_rig.center()))

            lower = lyap_rig.lower()
            upper = lyap_rig.upper()
            print
            scale = 1.0 if not hasattr(D,'_scale') else RR(D._scale.center())
            print 'Noise: ', coarse_noise_abs,'/',Kcoarse,'*',\
                   str(scale).rstrip('0'),'=',noise
            print 'Measure L1 err: ',rigerr[0],' (a priori: ',rigerr[1],')'
            print 'Measure L1 err: ',expnot(rigerr[0]),' (a priori: ',expnot(rigerr[1]),')'
            print 'Lyapunov Exp: [',lower,',',upper,']'
            print 'Lyapunov Exp: ',expnotint(lyap_rig)

            myplot += line([(noise*0.98, upper+0.008), (noise, upper),
                            (noise/0.98, upper+0.008)],
                 color="#f00")
            myplot += line([(noise*0.98, lower-0.008), (noise, lower),
                            (noise/0.98, lower-0.008)],
                 color="#00f")
        else:
            print('Not found: "{0}"'.format(riglyap_file))
    data_file.close()
    myplot.show(**plot_args)
