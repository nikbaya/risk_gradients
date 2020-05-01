#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 11:41:43 2020

@author: nbaya
"""
#
#import hail as hl
#from hail.linalg import BlockMatrix
#
#hl.init(log='/tmp/hail.log')

import numpy as np

for M in [100]:
    
    for N in [10000]: # np.logspace(3,5,3, dtype='int')
    
#        f = np.random.uniform(0.05,0.95,size=M) # allele freqs
#        X = np.random.binomial(n=2, p=0.5, size=(N, M)).astype('float')
        X = np.random.normal(size=(N,M))
        X -= X.mean(axis=0)
        X /= X.std(axis=0)
        
        X_new = X.copy()        
        
        for i in range(10):
            
            X = X_new.copy()
            
            R = (1/N)*X.T@X
            L = np.linalg.cholesky(R)
            
            L_inv = np.linalg.inv(L)
            
            X_alt = X@L_inv
            
            R_alt = (1/N)*X_alt.T@X_alt
        
        
        def E():
            Z = np.random.normal(0,1,size=N)
            return (1/np.sqrt(N))*(X.T)@Z
        
        for h2 in [0.5]: #np.linspace(0.1, 0.5, 4):

            # infinitesimal
            alpha = np.random.normal(loc=0, scale=np.sqrt(h2/M), size=M)

            # spike & slab
#            pi = 0.01
#            alpha = np.random.normal(loc=0, scale=np.sqrt(h2/(pi*M)), size=M)
            
            yg = X@alpha
            
            s2 = (1/N)*(yg.T)@(yg)
            
            alpha *= np.sqrt(h2/s2) # comment out later?
            
            beta = (1/N)*X.T@yg
            
            print(f'var(alpha)*M/h2 = {alpha.var()*M/h2}')
            print(f'var(beta)*M/h2 = {beta.var()*M/h2}')
            
            print(f'corr(alpha, beta) = {np.corrcoef(alpha, beta)[0,1]}')
            
            # GWAS sumstats
            for N_d in [1000000]: #np.logspace(3,6,4):
                betahat = beta + (1/np.sqrt(N_d))*E()
                
                alphahat = betahat.copy()
                
                noise = np.random.normal(loc=0,scale=np.sqrt(1-h2),size=N)
                noise *= np.sqrt(1-h2)/noise.std() # comment out later?
                
                y = yg + noise
                
#                print(f'var(yg): {yg.var()}')
#                print(f'var(y): {y.var()}')
                
                yg_hat = X@alphahat
                
                # "true" r2 when calculating 
                r2_true = np.corrcoef(y,yg_hat)[0,1]**2
                r2_summ = (((1/N)*(yg.T)@(yg_hat))**2)/((1/N)*yg_hat.T@yg_hat)
                
                
                print(f'r2_true: {r2_true}')
                print(f'r2_summ: {r2_summ}')
                
                print(f'r2_daet: {h2/(1+M/(N_d*h2))}')
                
#            
#            

if __name__=='__main__':
    gs_bucket= 'gs://nbaya/risk_gradients'

    ref_panel = '1kg_eur'
    
    X = hl.linalg.BlockMatrix.read(f'{gs_bucket}/{ref_panel}.bm')
    X = X.T
    
    N = X.shape[0]
    M = X.shape[1]
    
#    R = (1/N)*X.T@X
#    R_bm_path = f'{gs_bucket}/{ref_panel}.R.bm'
#    if not hl.hadoop_is_file(f'{R_bm_path}/_SUCCESS'):
#        R.write(R_bm_path)
#    R = BlockMatrix.read(R_bm_path)
    
    Z = np.random.multivariate_normal(mean=np.zeros(shape=N),
                                      cov=np.identity(n=N))
    Z = hl.linalg.BlockMatrix.from_numpy(Z).T
    
    E = 1/np.sqrt(N)*X.T@Z
    
    E_bm_path = f'{gs_bucket}/{ref_panel}.E.bm'
    if not hl.hadoop_is_file(f'{E_bm_path}/_SUCCESS'):
        E.write(E_bm_path)
    E = BlockMatrix.read(E_bm_path)
        
    seed = 1
    np.random.seed(seed=seed)
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
    

    
    
    
    
    
    
    
    
    
    
    