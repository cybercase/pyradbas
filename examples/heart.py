#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:  Stefano Brilli
# Date:    1/11/2011
# E-mail:  stefanobrilli@gmail.com

import numpy as np
import matplotlib.pyplot as plt
import pyradbas as pyrb

#Defines the mesh
x = np.linspace(-1.5, 1.5, 40)
x, y = np.meshgrid(x, x)
P = np.vstack([x.flatten(), y.flatten()])

#Defines the function
heart = lambda x,y: np.abs(x**2.+2*(y-0.5*np.sqrt(np.abs(x)))**2.-1)

#Evaluates function over every point of the grid
V = heart(P[0:1], P[1:])

#Plot
plt.plot(P[0:1][V<0.2], P[1:][V<0.2], '*r', P[0:1][V>=0.2], P[1:][V>=0.2], 'o')
plt.show()
#defines an exact RBFN
enet = pyrb.train_exact(P.T, V.T, 0.3)

#simulate
S = enet.sim(P.T).T
plt.plot(P[0:1][S<0.2], P[1:][S<0.2], '*r', P[0:1][S>=0.2], P[1:][S>=0.2], 'o')
plt.show() # small differences are due to ill conditioning
#What if we compute points outside training set
O=np.random.uniform(size=(2,5000), low=-2., high=2.)
S = enet.sim(O.T).T
plt.plot(O[0:1][S<0.2], O[1:][S<0.2], '*r', O[0:1][S>=0.2], O[1:][S>=0.2], 'o')
plt.show()
#To achieve better generalization we can train an inexact RBFN
inet = pyrb.train_ols(P.T, V.T, 0.0007, 0.3, verbose=True)

#simulate
S = inet.sim(P.T).T
plt.plot(P[0:1][S<0.2], P[1:][S<0.2], '*r', P[0:1][S>=0.2], P[1:][S>=0.2], 'o')
plt.show()
#Outside training set...
S = inet.sim(O.T).T
plt.plot(O[0:1][S<0.2], O[1:][S<0.2], '*r', O[0:1][S>=0.2], O[1:][S>=0.2], 'o')
plt.show()

