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
        v = np.exp( -(np.sqrt(((np.atleast_2d(x)[:, np.newaxis]-self.centers[np.newaxis, :])**2.0).sum(-1))*self.ibias)**2.0 )
        v = np.dot(v, self.linw) + self.obias
        return v

