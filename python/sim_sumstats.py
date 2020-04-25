#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 11:41:43 2020

@author: nbaya
"""

import numpy as np

M = 1000


# reference
N_ref = 100

Z = np.random.multivariate_normal(mean=np.zeros(shape=N_ref), 
                                  cov=np.identity(N_ref))

X = np.random.binomial(n=2, p=0.5, size=(N_ref, M)).astype('float')
X -= X.mean(axis=0)
X /= X.std(axis=0)

E = (1/np.sqrt(N_ref))*(X.T)@Z

R = (1/N_ref)*X.T@X

h2 = 0.2
# infinitesimal
alpha = np.random.normal(loc=0, scale=np.sqrt(h2/M), size=M)

beta = R@alpha



plt.hist(E)
