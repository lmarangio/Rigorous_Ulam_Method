"""
Functions to work with intervals (only Newton remained for now, there used to be other stuff but it's obsolete)
"""

__all__ = ["interval_newton,NewtonException"]

def interval_newton(f, fprime, I, alpha, epsilon):
    """
    Finds an interval of width epsilon where f(x)-alpha has a zero. f is supposed to be monotonic.
    f and fprime must take and return intervals.
    If alpha is not contained in f(I), returns [I.lower(), I.lower()] or [I.upper(), I.upper()]
    """
    origI = I
    intervalfield = I.parent()
    
    alpha = intervalfield(alpha)
    
    iterations = 0
    while I.absolute_diameter() > epsilon:
        #print I.lower(), I.upper()
        iterations += 1
        if iterations > 100:
            print 'branch:', origI.lower(),origI.upper()
            print 'value:', alpha 
            raise ValueError, 'Interval Newton did not converge. Use higher precision?'
        m = intervalfield(I.center())
        fprimeI = fprime(I)
        delta = f(m)-alpha
        if delta.lower() < 0 and delta.upper() > 0:
            m = intervalfield((3*I.lower()+I.upper())/4)
            delta = f(m)-alpha
            if delta.lower() < 0 and delta.upper() > 0:
                m = intervalfield((I.lower()+4*I.upper())/4)
                delta = f(m)-alpha
        ratio = delta/fprimeI
        intwith = m - ratio
        #print 'fprimeI',fprimeI.lower(), fprimeI.upper()
        #print 'fprimeIinv',fprimeIinv.lower(), fprimeIinv.upper()
        #print 'delta',delta.lower(), delta.upper()
        #print 'ratio',ratio.lower(), ratio.upper()
        #print 'intwith',intwith.lower(), intwith.upper()
        if intwith.lower() >= I.upper():
            return intervalfield(origI.upper())
        if intwith.upper() <= I.lower():
            return intervalfield(origI.lower())
        I = I.intersection(intwith)
        #print 'I',I.lower(), I.upper()
    return I
