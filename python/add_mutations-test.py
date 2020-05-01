#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 13:33:31 2020

Add mutations to tree sequence

@author: nbaya
"""


import msprime
import tskit

seed = 1

ts0 = tskit.load(f'n_23000.rec_2e-08.mut_2e-08.seed_{seed}.m_51229805.chr_22.ts')

print(f'init num mutations: {ts0.get_num_mutations()}')

mut = 2e-8
mut_model = msprime.InfiniteSites(alphabet=0) # to use nucleotides: alphabet=msprime.NUCLEOTIDES
keep = False #whether to keep existing sites and mutations in tree sequence

ts = msprime.mutate(tree_sequence=ts0,
                    rate=mut,
                    model=mut_model,
                    keep=keep)


print(f'final num mutations: {ts.get_num_mutations()}')

