from assembler import assemble
from dynamic import QuotientedLinearDynamic, PerturbedFourxDynamic, MannevillePomeauDynamic
from partition import equispaced, partition_diameter

from sage.all import *

prec = 256
epsilon = 1e-23;

# sample homogeneous partition
k = 8;
hompartition = equispaced(2**k)

# sample non-homogeneous partition: width 1/2^(k+1) up to 0.5, 1/2^(k) afterwards
k = 8;
nonhompartition = [i*(0.5 ** (k+1)) for i in range(2 ** k)] + [0.5 + i*(0.5 ** k) for i in range(2 ** (k-1)+1)]

k = 13;
largepartition = [i*(0.5 ** k) for i in range(2**k + 1)]

print "Computing dynamic with %d digits and epsilon=%g" % (prec, epsilon)

D = PerturbedFourxDynamic(prec=prec)

P = assemble(D, nonhompartition, epsilon=epsilon, prec=prec)

e = P.parent().row_space().dense_module()([1] * P.dimensions()[0])
infnorm = max(x.magnitude() for x in e*P-e)
print "Stochasticity error: %g" % infnorm
print "(should be smaller than epsilon/delta = %g)" % (epsilon / partition_diameter(nonhompartition))
assert infnorm < epsilon / partition_diameter(nonhompartition)

D = MannevillePomeauDynamic(prec=prec)
P = assemble(D, nonhompartition, epsilon=epsilon, prec=prec)

e = P.parent().row_space().dense_module()([1] * P.dimensions()[0])
infnorm = max(x.magnitude() for x in e*P-e)
print "Stochasticity error: %g" % infnorm
print "(should be smaller than epsilon/delta = %g)" % (epsilon / partition_diameter(nonhompartition))
assert infnorm < epsilon / partition_diameter(nonhompartition)

D = QuotientedLinearDynamic(4, prec=prec)
P = assemble(D, hompartition, epsilon=epsilon, prec=prec)

e = P.parent().row_space().dense_module()([1] * P.dimensions()[0])
infnorm = max(x.magnitude() for x in e*P-e)
print "Stochasticity error: %g" % infnorm
print "(should be smaller than epsilon/delta = %g)" % (epsilon / partition_diameter(hompartition))
assert infnorm < epsilon / partition_diameter(hompartition)

