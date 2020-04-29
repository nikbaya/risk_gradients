#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 11:41:43 2020

@author: nbaya
"""

import hail as hl
import numpy as np
from hail.linalg import BlockMatrix

hl.init(log='/tmp/hail.log')

#import matplotlib.pyplot as plt
#
##for M in np.logspace(3,4,2,dtype='int'):
#for M in [10000]:
#    
#    # reference
##    for N_ref in np.logspace(3,5,3, dtype='int'):
#    for N_ref in [1000]:
#        
#        Z = np.random.multivariate_normal(mean=np.zeros(shape=N_ref), 
#                                          cov=np.identity(N_ref))
#    
#        f = np.random.uniform(0.05,0.95,size=M)
#        X = np.random.binomial(n=2, p=f, size=(N_ref, M)).astype('float')
#        X -= X.mean(axis=0)
#        X /= X.std(axis=0)
#        
#        E = (1/np.sqrt(N_ref))*(X.T)@Z
#        
#        R = (1/N_ref)*X.T@X
#        
#        for h2 in np.linspace(0.1, 0.5, 4):
##            h2 = 0.2
#            # infinitesimal
#            alpha = np.random.normal(loc=0, scale=np.sqrt(h2/M), size=M)
#            
#            # spike & slab
##            pi = 0.01
##            alpha = np.random.normal(loc=0, scale=np.sqrt(h2/(pi*M)), size=M)
#            
#            beta = R@alpha
#            
#            # GWAS sumstats
#            for N_d in np.logspace(3,6,4):
##                N_d = 10000
#                betahat = beta + (1/np.sqrt(N_d))*E
#                alphahat = betahat
#                
#                r_beta = np.corrcoef(betahat, beta)[0,1]
#                r = h2*(alphahat.T@R@alpha)/np.sqrt((alpha.T@R@alpha)*(alphahat.T@R@alphahat))
#                
#                print('-------------')
#                print(f'M: {M}')
#                print(f'N_ref: {N_ref}\tN_d: {int(round(N_d,0))}')
#                print(f'h2: {h2}')
#                print(f'beta r2: {r_beta**2}')
#                print(f'r2: {r**2}')
#            
#            

if __name__=='__main__':
    gs_bucket= 'gs://nbaya/risk_gradients'

    ref_panel = '1kg_eur'
    
    X = hl.linalg.BlockMatrix.read(f'{gs_bucket}/{ref_panel}.bm')
    X = X.T
    
    N = X.shape[0]
    M = X.shape[1]
    
    R = (1/N)*X.T@X
    R_bm_path = f'{gs_bucket}/{ref_panel}.R.bm'
    if not hl.hadoop_is_file(f'{R_bm_path}/_SUCCESS'):
        R.write(R_bm_path)
    R = BlockMatrix.read(R_bm_path)
    
    Z = np.random.multivariate_normal(mean=np.zeros(shape=N),
                                      cov=np.identity(n=N))
    Z = hl.linalg.BlockMatrix.from_numpy(Z).T
    
    E = 1/np.sqrt(N)*X.T@Z
    
    E_bm_path = f'{gs_bucket}/{ref_panel}.E.bm'
    if not hl.hadoop_is_file(f'{E_bm_path}/_SUCCESS'):
        E.write(E_bm_path)
    E = BlockMatrix.read(E_bm_path)
        
    h2 = 0.2
    alpha = np.random.normal(loc=0, scale=np.sqrt(h2/M), size=M)
    alpha = BlockMatrix.from_numpy(alpha).T
    
    beta = R@alpha
    
    beta_bm_path = f'{gs_bucket}/{ref_panel}.beta.bm'
    if not hl.hadoop_is_file(f'{beta_bm_path}/_SUCCESS'):
        beta.write(beta_bm_path, overwrite=True)
    beta = BlockMatrix.read(beta_bm_path)
    
    N_d = 10000
    betahat = beta + (1/np.sqrt(N_d))*E
    
    betahat_bm_path = f'{gs_bucket}/{ref_panel}.betahat.bm'
    if not hl.hadoop_is_file(f'{betahat_bm_path}/_SUCCESS'):
        betahat.write(betahat_bm_path, overwrite=True)
    betahat = BlockMatrix.read(betahat_bm_path)
        
    
    alphahat = betahat
    r_prs = h2*(alphahat.T@R@alpha)/np.sqrt((alphahat.T@R@alphahat)*(alpha.T@R@alpha))
    
    
    tb_r_prs = r_prs.to_table_row_major()
    tb_r_prs.show()
    
#    r_prs = r_prs.checkpoint(f'{gs_bucket}/tmp.1kg_eur.r_prs.bm',
#                             overwrite=True)
    

    
    
    
    
    
    
    
    
    
    
    