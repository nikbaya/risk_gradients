#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 11:41:43 2020

@author: nbaya
"""

import numpy as np

for M in [100]:
    
    for N in [10000]: # np.logspace(3,5,3, dtype='int')
    
#        f = np.random.uniform(0.05,0.95,size=M) # allele freqs
#        X = np.random.binomial(n=2, p=0.5, size=(N, M)).astype('float')
        X = np.random.normal(size=(N,M))
        X -= X.mean(axis=0)
        X /= X.std(axis=0)
        
        X_new = X
        
        triu_idx = np.triu_indices(n=N, k=1,m=M)
        diag_idx = np.diag_indices(n=M)

        R = (1/N)*X.T@X        
        mean_var = R[diag_idx].mean()
        mean_cov = R[triu_idx].mean()
        
#        print(f'mean var: {mean_var}')
#        print(f'mean cov: {mean_cov}')

        for i in range(5):
            
            X = X_new.copy()
            
            R = (1/N)*X.T@X
            L = np.linalg.cholesky(R)
            
            L_inv = np.linalg.inv(L)
            
            X_new = X@L_inv
            
            R_new = (1/N)*X_new.T@X_new
            
#            print(R_new[:3,:3])
            
            mean_var = R_new[diag_idx].mean()
            mean_cov = R_new[triu_idx].mean()
            
#            print(f'mean var: {mean_var}')
#            print(f'mean cov: {mean_cov}')
            
        X = X_new
        
        
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