# -*- coding: utf-8 -*-
# Author:  Stefano Brilli
# Date:    24/10/2011
# E-mail:  stefanobrilli@gmail.com

import rbfn
import numpy as np
import numpy.linalg as la

def train_ols(I, O, mse, gw=1.0, verbose=False):
    """
    Build a rbfn
    I (N by M) N vector of M size
    O (N by T) N vector of T size
    """
    k = np.sqrt(-np.log(0.5))/gw
    m, d = O.shape
    d *= m
    idx = np.arange(m)
    P = np.exp(-( np.sqrt(((I[np.newaxis,:] - I[:, np.newaxis])**2.0).sum(-1)) * k)**2.0)
    G = np.array(P)
    D = (O*O).sum(0)
    e = ( ((np.dot(P.T, O)**2.) / ((P*P).sum(0)[:,np.newaxis]*D) )**2.).sum(1)
    next = e.argmax()
    used = np.array([next])
    idx = np.delete(idx, next)
    W = P[:,next, np.newaxis]
    P = np.delete(P, next, 1)
    G1 = G[:, used]
    t, r, _, _ = la.lstsq(G1, O)
    err = r.sum()/d
    while err > mse and P.shape[1] > 0:
        if verbose:
            print err, m-P.shape[1]
        wj = W[:, -1:]
        a = np.dot(wj.T, P)/np.dot(wj.T, wj)
        P = P-wj*a
        e = (((np.dot(P.T, O)**2.) / ((P*P).sum(0)[:,np.newaxis]*D) )**2.).sum(1)
        next = e.argmax()
        W = np.append(W, P[:,next, np.newaxis], axis=1)
        used = np.append(used, idx[next])
        P = np.delete(P, next, 1)
        idx = np.delete(idx,next)
        t, r, _, _ = la.lstsq(G[:, used], O)
        err = r.sum()/d
    if verbose:
        print err, m-P.shape[1]
    net = rbfn.Rbfn(centers=I[used], linw=t, ibias=k, obias=0.)
    return net

if __name__ == "__main__":
    # Simple test: recognising of points inside a ring
    N = 1000
    I = (np.random.uniform(size=(N,2), low=-1., high=1.))
    O = np.zeros((N,1))
    O[ ((I**2.).sum(1) < 1)*((I**2.).sum(1) > 0.5)] = 1.0
    r = train_ols(I, O, 0.03, 0.27, True)

    # Plot of some test value
    import matplotlib.pyplot as plt
    T = (np.random.uniform(size=(N*5,2), low=-1., high=1.))
    V = r.sim(T).flatten()
    OUT = T[V<0.5].T
    IN = T[V>=0.5].T
    plt.plot(IN[0], IN[1], 'r*', OUT[0], OUT[1], 'bo')
    plt.show()

