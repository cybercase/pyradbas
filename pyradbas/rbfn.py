# -*- coding: utf-8 -*-
# Author:  Stefano Brilli
# Date:    24/10/2011
# E-mail:  stefanobrilli@gmail.com

import numpy as np
class Rbfn(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def sim(self, x):
        """
        Run the network over a single input and return the output value
        """
        v = np.atleast_2d(x)[:, np.newaxis]-self.centers[np.newaxis, :]
        v = np.sqrt( (v**2.).sum(-1) ) * self.ibias
        v = np.exp( -v**2. )
        v = np.dot(v, self.linw) + self.obias
        return v

    def _input_size(self):
        try:
            return self.centers.shape[1]
        except AttributeError:
            return -1
    input_size = property(_input_size)

    def _output_size(self):
        try:
            return self.linw.shape[1]
        except AttributeError:
            return -1
    output_size = property(_output_size)
