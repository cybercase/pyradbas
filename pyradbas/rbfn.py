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

    def _input_size(self):
        try:
            return self.centers.shape[1]
        except:
            return -1
    input_size = property(_input_size)

    def _output_size(self):
        try:
            return self.linw.shape[1]
        except:
            return -1
    output_size = property(_output_size)
