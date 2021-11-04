
from sage.all import *
from interval import interval_newton

def bisect_find(f, I, alpha, epsilon):
    RI = parent(I)
    interval = I
    l,u = RI(I.lower()), RI(I.upper())
    fa = f(l)-alpha
    fb = f(u)-alpha

    while true:
        retv = l.union(u)
        #print([retv,retv.diameter()])
        if retv.diameter() < epsilon:
            return retv
        m = RI(((l+u)/2).center())
        fm = f(m)-alpha
        if fm*fa > 0:
            l,fa = m,fm
        else:
            u,fb = m,fm

def safe_interval_newton(f, fprime, I, alpha, epsilon):
    RI = parent(I)
    interval = I
    fa = f(RI(I.lower()))-alpha
    fb = f(RI(I.upper()))-alpha

    while true:
        if interval.diameter() < epsilon:
            return interval

        derint = fprime(interval)

        if ((derint.lower() > 0 and derint.lower()/derint.upper() > 0.2) or
           (derint.upper() < 0 and derint.upper()/derint.lower() > 0.2)):
           return interval_newton(f, fprime, interval, alpha, epsilon)

        #bisect
        c = interval.center()
        fmid = f(RI(c))-alpha
        if fmid.upper() <= 0 and fa.lower() >= 0:
            fb = fmid
            interval = RI(interval.lower(), c)
            continue
        if fmid.upper() <= 0 and fb.lower() >= 0:
            fa = fmid
            interval = RI(c, interval.upper())
            continue
        if fmid.lower() >= 0 and fa.upper() <= 0:
            fb = fmid
            interval = RI(interval.lower(), c)
            continue
        if fmid.lower() >= 0 and fb.upper() <= 0:
            fa = fmid
            interval = RI(c, interval.upper())
            continue
        if fa.upper() <= 0 and fb.upper() <= 0:
            return RI(I.upper()) if fb.upper() > fa.upper() else RI(I.lower())
        if fa.lower() >= 0 and fb.lower() >= 0:
            return RI(I.lower()) if fb.lower() > fa.lower() else RI(I.upper())
        print 'alpha',alpha
        print 'iv',interval.lower(), interval.upper()
        print 'ivdiam',interval.diameter()
        print 'eps',epsilon
        print 'fmid',fmid.lower(), fmid.upper()
        print 'fa',fa.lower(), fa.upper()
        print 'fb',fb.lower(), fb.upper()
        raise ValueError, "Can't go on with bisection!"
