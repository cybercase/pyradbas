#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:  Stefano Brilli
# Date:    24/10/2011
# E-mail:  stefanobrilli@gmail.com

# From project root exec: PYTHONPATH=. python examples/matlab.py
# The example network is trained with a random set of 1500 2D points. Output
# is 1 if the training point is inside the ring of minimum and maximum radius
# respectively of 0.2 and 0.5. Otherwise is 0

from __future__ import print_function
import pyradbas as pyrb
import numpy as np
import matplotlib.pyplot as plt

samples = 5000
net = pyrb.mlab.load("examples")
T = np.random.uniform(size=(samples, 2), low=-1.0, high=1.0)

print("Running...")
D = net.sim(T).flatten()
print("Done.")

IN = T[D >= 0.5]
OUT = T[D < 0.5]

print("Plotting... ", end='')
plt.plot(IN[:,0], IN[:,1], 'r*', OUT[:,0], OUT[:,1], 'bo')
print("Done.\nClose the plot window to exit.")
plt.show()
