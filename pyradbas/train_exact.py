# -*- coding: utf-8 -*-
# Author:  Stefano Brilli
# Date:    24/10/2011
# E-mail:  stefanobrilli@gmail.com

import rbfn
import numpy as np
import numpy.linalg as la

def train_exact(I, O, gw=1.0):
    """
    Build an exact (zero error for inputs) radial basis network
    I (N by M) N vector of M size
    O (N by T) N vector of T size
    """
    k = np.sqrt(-np.log(0.5))/gw
    G = ((I[np.newaxis,:,:] - I[:, np.newaxis, :])**2.).sum(-1)
    G = np.exp(-( np.sqrt(G)*k )**2.0)
    W = la.lstsq(G,O)[0]
    net = rbfn.Rbfn(centers=I, ibias=k, linw=W, obias=0)
    return net

if __name__ == "__main__":
    # Simple test: recognising of points inside a ring
    N = 1000
    I = (np.random.uniform(size=(N,2))-0.5)*2.
    O = np.zeros((N,1))
    O[ ((I**2.).sum(1) < 1)*((I**2.).sum(1) > 0.5)] = 1.0
    r = train_exact(I, O, 0.27)
    err = abs(r.sim(I) - O)

    # The next error should be as much as possible closer to zero. Higher values
    # may occur due to ill conditioned problems.
    print "Error: ", max(np.sqrt((err**2.).sum(1)))
    # Plot of some test value
    import matplotlib.pyplot as plt
    T = (np.random.uniform(size=(N*5,2))-0.5)*2
    V = r.sim(T).flatten()
    OUT = T[V<0.5].T
    IN = T[V>=0.5].T
    plt.plot(IN[0], IN[1], 'r*', OUT[0], OUT[1], 'bo')
    plt.show()
