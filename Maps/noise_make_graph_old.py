
import numpy as np
import os
import joblib

from bzmodel import *
from noise_contraction_proof import *

prec = 256
D = BZModel(prec)
RI = D.field

# same parameter set as in noise_induced_order.py
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
    #(D, 2**26, 2**20, 70, 136,  2**17),  #0.0001296
    #(D, 2**26, 2**20, 70, 112,  2**17),  #0.0001068
]

def expnot(num, dig = 3):
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

myplot = plot([])

data_file = file('data_sheet.tex', 'w')
header = ['delta','delta_contr','delta_est','noise','n_contr','alpha_contr','alpha',
                  'sumCi', 'l1apriori', 'l1err', 'lyap']
data_file.write("\t".join(header)+"\n")

for D, K, Kcoarse, num_iter, coarse_noise_abs, Kest in params:
    fingerprint = 'P%d_K%d_C%d_N%d_I%d_E%d' % (D.field.prec(), K, Kcoarse,
                                           coarse_noise_abs, num_iter, Kest)
    rigerr_file = 'bzmodel_results/rigerr_' + fingerprint
    riglyap_file = 'bzmodel_results/riglyap_' + fingerprint

    if os.access(riglyap_file, os.R_OK):
        estFactor, sumCi, alpha, N, alpha_contr = prove_contraction(D, Kcoarse,
                                                      coarse_noise_abs, num_iter)
        
        rigerr = joblib.load(rigerr_file)
        lyap_rig,li,ui = joblib.load(riglyap_file)
        noise = RR(coarse_noise_abs)/Kcoarse*RR(D._scale.center())

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
        
        lower = lyap_rig.lower()
        upper = lyap_rig.upper()
        print
        print 'Noise: ', coarse_noise_abs,'/',Kcoarse,'*',\
               str(D._scale.center()).rstrip('0'),'=',noise
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


data_file.close()
myplot.show(scale = "semilogx", xmin = 0.0001, xmax = 0.01, ymin = -0.65, ymax = 0.1)
