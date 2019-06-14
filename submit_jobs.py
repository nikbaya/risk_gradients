#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 09:58:02 2019

For submitting hard-parallelized jobs across clusters

@author: nbaya
"""

import subprocess
import os

phen = '50'
frac_all_ls = [0.2, 0.4, 0.6, 0.8, 1]
frac_cas_ls = [1]
parsplit = len(frac_all_ls) # number of parallel batches to run

os.chdir('/Users/nbaya/Documents/lab/risk_gradients/')

for i, frac in enumerate(frac_all_ls):
    start = f'cluster start ukbb-nb-{i} --max-idle 10m --num-workers 10'
    submit = f'cluster submit ukbb-nb-{i} calc_scores.py --args "--phen {phen} --frac_all_ls {frac}"'
    subprocess.call(start+' ; '+submit)
    
    subprocess.call(['cluster','list'])