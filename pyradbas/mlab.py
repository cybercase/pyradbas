# -*- coding: utf-8 -*-
# Author:  Stefano Brilli
# Date:    24/10/2011
# E-mail:  stefanobrilli@gmail.com

import numpy as np
import rbfn
import os

def export_code(rbfn_name="net", ibias_fn="ibias.net", obias_fn="obias.net", linw_fn="linw.net", centers_fn="centers.net"):
    """
    Return the matlab export code for a radial basis network
    """
    script = """%% Auto Generated m-file script
ibias = %(RBFN_NAME)s.b{1};
obias = %(RBFN_NAME)s.b{2};
linw = %(RBFN_NAME)s.lw{2};
centers = %(RBFN_NAME)s.iw{1};
save '%(IBIAS)s' ibias -ASCII -DOUBLE
save '%(OBIAS)s' obias -ASCII -DOUBLE
save '%(LINW)s' linw -ASCII -DOUBLE
save '%(CENTERS)s' centers -ASCII -DOUBLE
"""
    val = {'RBFN_NAME':rbfn_name, 'IBIAS':ibias_fn, 'OBIAS':obias_fn,
           'LINW': linw_fn, 'CENTERS':centers_fn}
    return script % val

def load(bpath=".",
         centers_fn="centers.net",
         ibias_fn="ibias.net",
         obias_fn="obias.net",
         linw_fn="linw.net",
         fmt = np.float64,
         sep="  "):
    """
    Create a Rbfn object from an exported matlab radial basis network.
    """
    ibias = np.fromfile(os.path.join(bpath, ibias_fn), dtype=fmt, sep=sep)
    obias = np.fromfile(os.path.join(bpath, obias_fn), dtype=fmt, sep=sep)
    linw = np.fromfile(os.path.join(bpath,linw_fn), dtype=fmt, sep=sep)
    centers = np.fromfile(os.path.join(bpath,centers_fn), dtype=fmt, sep=sep)
    idim = centers.shape[0]/ibias.shape[0]
    cdim = ibias.shape[0]
    odim = obias.shape[0]
    r = rbfn.Rbfn()
    r.centers = centers.reshape(cdim, idim)
    r.ibias = ibias.reshape((cdim, 1)).T
    r.linw = linw.reshape((odim, cdim)).T
    r.obias = obias.reshape((odim, 1))
    return r

