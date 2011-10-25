# -*- coding: utf-8 -*-
# Author:  Stefano Brilli
# Date:    24/10/2011
# E-mail:  stefanobrilli@gmail.com

import numpy as np
class Rbfn(object):
    def __init__(self, *args, **kwargs):
        pass

    def sim(self, x):
        """
        Run the network over a single input and return the output value
        """
        cdim = self.centers.shape[0]
        v = np.sqrt(((x-self.centers)**2.0).sum(1)).reshape((cdim,1))*self.ibias
        v = np.exp(-v**2.0)
        v = np.dot(self.linw, v) + self.obias
        return v

