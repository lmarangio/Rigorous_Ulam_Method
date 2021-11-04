from bzmodel import *

D = BZModel(256)
RI = D.field

nlrg = RI(0.01)
xi = RI(166) / 2**20 # smallest noise considered
A = [ (RI(0.1249, 0.1251)-D._start)/D._scale, (RI(0.2999, 0.3001)-D._start)/D._scale ]
Anlrg = [(x+nlrg).union(x-nlrg).intersection(RI(0,1)) for x in A]

K = 2**15
partitions = [RI(i,(i+1))/K for i in range(K)]

#
# We will compute bounds for:
# 0. sup of |L1| in A_xi
# 1. sup of |log|T'|| outside A
# 2. integral of log|T'| inside A


sup_L1 = RI(0)
for xtr in partitions:
    if not any(xtr.overlaps(iv) for iv in Anlrg):
        continue
    cur_sum = RI(0)
    for i,iv in enumerate(D.branch_intervals):
        #print('i=',i)
        a = D.preimage(RI(xtr.lower()), i, 2**(-128))
        b = D.preimage(RI(xtr.upper()), i, 2**(-128))
        pre_xtr = a.union(b)
        cur_sum += (RI(1.0)/D.f_prime(pre_xtr)).abs()
    sup_L1 = sup_L1.max(cur_sum)

print('sup of |L1| on Anlrg:', sup_L1.upper())


sup_lg = RI(0)
for p in partitions:
    if any(p in iv for iv in A):
        continue
    sup_lg = sup_lg.max(D.f_prime(p).abs().log().abs())

print('Sup of |log|T\'|| outside A:', sup_lg.upper())

int_a_bound = (D.bound_int_p_sing1_log_f1_prime(RI(A[0].lower())).abs() +
                D.bound_int_sing1_q_log_f1_prime(RI(A[0].upper())).abs() +
                D.bound_int_p_fence_log_f1_prime(RI(A[1].lower())).abs() +
                D.bound_int_fence_q_log_f2_prime(RI(A[1].upper())).abs())

print('Integral of |log|T\'|| in A:', int_a_bound.upper())

invder_preAnlrg = RI(0)
distrt_preAnlrg = RI(0)

for p in partitions:
    if any(D.f(p).overlaps(iv) for iv in A):
        fp = D.f_prime(p)
        invder_preAnlrg = invder_preAnlrg.max((RI(1.0)/fp).abs())
        distrt_preAnlrg = distrt_preAnlrg.max((D.f_second(p)/(fp*fp)).abs())

print('sup of 1/T\' on T^-1(Anlrg) = invmu:', invder_preAnlrg.upper())
print('sup of T\'\'/(T\')^2 on T^-1(Anlrg) = Delta:', distrt_preAnlrg.upper())

sum_invT = RI(0)
for iv in D.branch_intervals:
    for xtr in iv:
        Txtr = D.f(xtr)
        if any(Txtr.overlaps(iv) for iv in Anlrg):
            sum_invT += (RI(1.0)/D.f_prime(xtr)).abs()

print('sum of |1/T\'| on boundary cap T^-1(Anlrg):', sum_invT)


#V_xi = 4/xi*invder_preAnlrg + distrt_preAnlrg + 2/xi*sum_invT
#print('V_xi:', V_xi.upper())
#print('for noise size:', xi, ' - in original scale:', xi*D._scale)
