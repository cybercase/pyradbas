# -*- coding: utf-8 -*-
# Author:  Stefano Brilli
# Date:    24/10/2011
# E-mail:  stefanobrilli@gmail.com

import numpy as np
class Rbn(object):
    def __init__(self, cdim, idim, odim):
        """
        cdim: number of centers
        idim: size of input vector
        odim: size of output vector
        """
        self.cdim = cdim
        self.idim = idim
        self.odim = odim

    def sim(self, x):
        """
        Run the network over a single input and return the output value
        """
        v = np.sqrt(((x-self.centers)**2.0).sum(1)).reshape((self.cdim,1))*self.ibias
        v = np.exp(-v**2.0)
        v = np.dot(self.linw, v) + self.obias
        return v

