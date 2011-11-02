#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:  Stefano Brilli
# Date:    2/11/2011
# E-mail:  stefanobrilli@gmail.com

from setuptools import setup, find_packages

setup(name="PyRadbas",
      version="0.1.0",
      description="Simple Radial Basis Function Networks in Python",
      long_description="""
      PyRadbas provides algorithms for building both exact and inexact RBFN.
      You can also use PyRadbas to import and run your Matlab(R) RBFN.
      """,
      keywords =['radial', 'basis', 'function', 'matlab', 'RBF', 'RBFN'],
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows'],
      zip_safe=True,
      author="Stefano Brilli",
      author_email="stefanobrilli@gmail.com",
      license = "FreeBSD",
      packages=find_packages(),
      url="http://cybercase.github.com/pyradbas/",
      install_requires=["numpy"],
      )

