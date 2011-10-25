# -*- coding: utf-8 -*-
# Author:  Stefano Brilli
# Date:    24/10/2011
# E-mail:  stefanobrilli@gmail.com

import rbfn
import numpy as np
import scipy.linalg as la2

def train_mse(I, O, mse, gw=1.0):
    """
    Build an exact (zero error for inputs) radial basis network
    I (N by M) N vector of M size
    O (N by T) N vector of T size
    """
    k = np.sqrt(-np.log(0.5))/gw
    patterns = zip(I,O)
    x, d = patterns.pop()
    centers = np.array([x])
    targets = np.array([d])
    G = np.array([[1.0]])
    W = np.hstack( [ la2.lstsq(G, o.T)[0].reshape(1,1) for o in targets.T ] )
    net = rbfn.Rbfn()
    net.centers = centers
    net.ibias = k
    net.linw = W.T
    net.obias = 0.0
    nextid = 0
    err = mse
    while patterns and err >= mse:
        errs = [ np.sqrt( ((net.sim(i)-o)**2.0).sum(0)) for i,o in patterns ]
        err = np.array(errs).sum()/len(errs)
        nextid = errs.index(max(errs))
        x, d = patterns.pop(nextid)
        #tmp = np.sqrt(np.vstack([ ((x-c)**2.0) for c in centers]).sum(1)).reshape(1,-1)
        tmp = np.exp( -( np.sqrt(np.vstack([ ((x-c)**2.0) for c in centers]).sum(1))*k )**2.0).reshape(1,-1)
        G = np.append(G, tmp.T, axis=1)
        G = np.append(G, np.append(tmp, np.array([[1.0]]), axis=1), axis=0 )
        centers = np.append(centers, [x], axis=0)
        targets = np.append(targets, [d], axis=0)
        W = np.hstack( [ la2.lstsq(G,o.T)[0] for o in targets.T ] )
        net.centers = centers
        net.ibias = k
        net.linw = W.T
        net.obias = 0.0
        print "Iteration", len(centers), "Error", err, "NID", nextid #, "Vrfy", max([np.sqrt(((net.sim(c)-d)**2.0).sum()) for c,d in zip(centers, targets)])
    return net

if __name__ == "__main__":
    # Simple test: recognising of points inside a ring
    # Obviusly, more are the points, better is the result
    N = 500
    I = (np.random.uniform(size=(N,2))-0.5)*2.
    O = np.zeros((N,1))
    O[ ((I**2.).sum(1) < 1)*((I**2.).sum(1) > 0.5)] = 1.0
    import time
    atime = time.time()
    r = train_mse(I, O, 0.03, 0.27)
    print time.time()-atime, "(s) elapsed"
    err = np.vstack([ abs(r.sim(x) - o) for x, o in zip(I,O) ])

    # The next error should be as much as possible closer to zero. Higher values
    # may occur due to ill conditioned problems.
    print "ERRORE", max(np.sqrt((err**2.).sum(1)))

    if 0:
        import sys
        sys.exit(0)
    # Plot of some test value
    import matplotlib.pyplot as plt
    T = (np.random.uniform(size=(N*5,2))-0.5)*2
    V = np.hstack([r.sim(t) for t in T]).reshape((N*5,))
    OUT = T[V<0.5].T
    IN = T[V>=0.5].T
    plt.plot(IN[0], IN[1], 'r*', OUT[0], OUT[1], 'bo')
    plt.show()
