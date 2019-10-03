#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 17:12:06 2019

Li-Stephens genotype simulation

Tim's suggestion
1. `it` is table of intervals, 
   `Nref` is how many reference individuals, 
   `Nsim` is how many to simulate, 
   `mt` is matrix table with reference genotypes
2. annotate `it` with an array of random numbers from 0 to `Nref`, of size `Nsim`, call this `a`
3. take `mt`, localize_entries to get a table with an array, call this `ht`
4. Annotate `ht` with `it`
5. map over `a`, take `ht.entries[i].GT`

@author: nbaya
"""



import hail as hl
import numpy as np



def ls_sim(ref_mt, Nsim, mu=0.01, Nref=None, snps=None):
    assert (mu > 0) and (mu < 1), 'Mutation rate (recombination rate) must be in (0,1)'
    assert 'locus' in ref_mt.row, '`locus` row field must be in `ref_mt`'

    Nref = ref_mt.count_cols() if Nref is None else Nref
    snps = ref_mt['locus'] if snps is None else snps
    
    ## Get intervals of non-recombination
    recomb_idx0 = np.random.binomial(1,mu,size=len(snps)-2) #subtract 2 from length because recombination events can't occur at first or last SNP
    recomb_idx1 = np.insert(arr=recomb_idx0,obj=0,values=1)
    recomb_idx2 = np.append(arr=recomb_idx1,values=1)
    recomb_idx3 = recomb_idx2==1
    recomb_ls = np.asarray(snps)[recomb_idx3]
    interval_ls = list(zip(recomb_ls[:-1], recomb_ls[1:]))
    print(f'number of recombination events: {len(interval_ls)-1}')

    ## get table of intervals
    it0 = hl.utils.range_table(len(interval_ls))
    intervals = hl.literal(interval_ls)
    it0 = it0.key_by(intervals = hl.interval(intervals[it0.idx][0], intervals[it0.idx][1]))
    it0.describe()
    
    ## simulation

    it = it0.annotate(hap = hl.range(Nsim).map(lambda i: [hl.int(hl.rand_unif(0, Nref)), 
                                                          hl.int(hl.rand_unif(0, Nref))]))
    ht = mt.localize_entries(entries_array_field_name='entries')
    ht = ht.annotate(it = it[ht.locus])
    ht = ht.annotate(sim_gt = ht.it.hap.map(lambda i: hl.parse_call(
            hl.str(ht.entries[i[0]].GT[0])+'|'+hl.str(ht.entries[i[1]].GT[1]))))
    ht = ht.annotate_globals(cols = hl.range(Nsim).map(lambda i: hl.struct(col_idx=i)))
    ht = ht.annotate(sim_gt = ht.sim_gt.map(lambda x: hl.struct(GT=x)))
    mt_sim = ht._unlocalize_entries('sim_gt','cols',['col_idx'])
    return mt_sim

##reference genotypes
mt = hl.methods.balding_nichols_model(2,2,100)

snps = mt.locus.collect()

## Get intervals of non-recombination
recomb_rate = 0.1
recomb_idx0 = np.random.binomial(1,recomb_rate,size=len(snps)-2) #subtract 2 from length because recombination events can't occur at first or last SNP
recomb_idx1 = np.insert(arr=recomb_idx0,obj=0,values=1)
recomb_idx2 = np.append(arr=recomb_idx1,values=1)
recomb_idx3 = recomb_idx2==1
recomb_ls = np.asarray(snps)[recomb_idx3]
interval_ls = list(zip(recomb_ls[:-1], recomb_ls[1:]))

## get table of intervals
it0 = hl.utils.range_table(len(interval_ls))
intervals = hl.literal(interval_ls)
it0 = it0.key_by(intervals = hl.interval(intervals[it0.idx][0], intervals[it0.idx][1]))
it0.describe()

#it = mt.rows()
#it = it.annotate(recomb = hl.rand_bool(recomb_rate))
#it = it.filter(it.recomb == 1)
#prev_array = hl.scan.take(it.locus, 1)
#it = it.annotate(interval = hl.or_missing(hl.len(prev_array) == 1, hl.interval(it.locus, prev_array[0])))
#it.show()





## simulation
Nref = mt.count_cols()
Nsim = 200

it = it0.annotate(hap = hl.range(Nsim).map(lambda i: [hl.int(hl.rand_unif(0, Nref)), 
                                                      hl.int(hl.rand_unif(0, Nref))]))

ht = mt.localize_entries(entries_array_field_name='entries')

ht = ht.annotate(it = it[ht.locus])

ht = ht.annotate(sim_gt = ht.it.hap.map(lambda i: hl.parse_call(
        hl.str(ht.entries[i[0]].GT[0])+'|'+hl.str(ht.entries[i[1]].GT[1]))))

ht = ht.annotate_globals(cols = hl.range(Nsim).map(lambda i: hl.struct(col_idx=i)))

ht = ht.annotate(sim_gt = ht.sim_gt.map(lambda x: hl.struct(GT=x)))

mt_sim = ht._unlocalize_entries('sim_gt','cols',['col_idx'])

sim = it.hap.collect()

mt_sim.GT.show(20)

mt.GT.show(20)

interval_ls[:5]





#
#interval_ls
#
#for start, stop in interval_ls:
#    
#    mt1 = mt1.annotate_entries(sim_gt = hl.or_missing(mt1.sample_idx==ids[0],mt1.GT))
    
    




