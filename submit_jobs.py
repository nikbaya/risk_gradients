#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 09:58:02 2019

For submitting hard-parallelized jobs across clusters

@author: nbaya
"""

import subprocess
import os

frac_ls = [0.2, 0.4, 0.6, 0.8, 1]
parsplit = len(frac_ls) # number of parallel batches to run

os.chdir('/Users/nbaya/Documents/risk_gradients/')
for i, frac in enumerate(frac_ls):
    start = ['cluster','start',f'ukbb-nb-{i}','--max-idle','10m','--num-workers','10']
    submit = ['cluster','submit',f'ukbb-nb-{i}','--args','"--"','--num-workers','10']
    subprocess.call()