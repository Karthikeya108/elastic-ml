from pysgpp import *

def cg(b, alpha, imax, epsilon, ApplyMatrix, reuse = False, verbose=True, max_threshold=None):
    if verbose:
        print "Starting Conjugated Gradient"
 
    epsilon2 = epsilon*epsilon
    
    i = 0
    temp = DataVector(len(alpha))
    q = DataVector(len(alpha))
    delta_0 = 0.0
    
    # calculate residuum
    if reuse:
        q.setAll(0)
        ApplyMatrix(q, temp)
        r = DataVector(b)
        r.sub(temp)
        delta_0 = r.dotProduct(r)*epsilon2
    else:
        alpha.setAll(0)

    ApplyMatrix(alpha, temp)
    r = DataVector(b)
    r.sub(temp)

    # delta
    d = DataVector(r)
    
    delta_old = 0.0
    delta_new = r.dotProduct(r)

    if not reuse:
        delta_0 = delta_new*epsilon2
    
    if verbose:
        print "Starting norm of residuum: %g" % (delta_0/epsilon2)
        print "Target norm:               %g" % (delta_0)

    while (i < imax) and (delta_new > delta_0) and (max_threshold == None or delta_new > max_threshold):
        # q = A*d
        ApplyMatrix(d, q)
        # a = d_new / d.q
        a = delta_new/d.dotProduct(q)

        # x = x + a*d
        alpha.axpy(a, d)

        if i % 50 == 0:
        # r = b - A*x
            ApplyMatrix(alpha, temp)
            r.copyFrom(b)
            r.sub(temp)
        else:
            # r = r - a*q
            r.axpy(-a, q)
        
        delta_old = delta_new
        delta_new = r.dotProduct(r)
        beta = delta_new/delta_old
        
        if verbose:
            print "delta: %g" % delta_new
        
        d.mult(beta)
        d.add(r)
        
        i += 1
        
    if verbose:    
        print "Number of iterations: %d (max. %d)" % (i, imax)
        print "Final norm of residuum: %g" % delta_new
    
    return (i,delta_new)

