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
from functools import reduce
#import numpy as np

def get_mt(chr_ls):
    variants = hl.import_table('gs://nbaya/split/hapmap3_variants.tsv')
    variants = variants.annotate(**hl.parse_variant(variants.v))
    variants = variants.key_by('locus','alleles') 
    gt0 = hl.import_bgen(path='gs://fc-7d5088b4-7673-45b5-95c2-17ae00a04183/imputed/ukb_imp_chr'+str(set(chr_ls)).replace(' ','')+'_v3.bgen',
                         entry_fields=['GT'],
                         n_partitions = 1000,
                         sample_file = 'gs://ukb31063/ukb31063.autosomes.sample',
                         variants=variants)
    return gt0

def ls_sim(ref_mt, Nsim, mu=0.01, Nref=None, recombs=None, Nnsps=None):
    '''
    Simulates `Nsim` new individuals based on the genotypes in `ref_mt`.
    
    '''
    assert (mu > 0) and (mu < 1), 'Mutation rate (recombination rate) must be in (0,1)'
    assert 'locus' in ref_mt.row, '`locus` row field must be in `ref_mt`'

    Nref = ref_mt.count_cols() if Nref is None else Nref

    ## get table of recombination hotspots
    it0 = ref_mt.rows().key_by('locus')    
    it0 = it0.add_index('tmp_idx')
    Nsnps = it0.count() 
    if recombs is None:
        it0 = it0.annotate(recomb=(hl.rand_bool(mu))|(it0.tmp_idx==0)|(it0.tmp_idx==Nsnps-1))
        recombs = it0.filter(it0.recomb==1).locus.collect()
    intervals = list(map(lambda start, stop: hl.interval(start, stop), recombs[:-1], recombs[1:]))
    intervals = intervals[:-1] + [hl.interval(intervals[-2].end, intervals[-1].end, includes_end=True)]
    it0 = hl.Table.parallelize([hl.struct(interval=interval) for interval in intervals])
    it0 = it0.key_by('interval')
    
    ## simulation
    it = it0.annotate(hap = hl.range(Nsim).map(lambda i: [hl.int(hl.rand_unif(0, Nref)), 
                                                          hl.int(hl.rand_unif(0, Nref))]))
    ht = ref_mt.localize_entries(entries_array_field_name='entries')
    ht = ht.annotate(it = it[ht.locus])
    ht = ht.annotate(sim_gt = ht.it.hap.map(lambda i: hl.parse_call(
            hl.str(ht.entries[i[0]].GT[0])+'|'+hl.str(ht.entries[i[1]].GT[1]))))
    ht = ht.annotate_globals(cols = hl.range(Nsim).map(lambda i: hl.struct(col_idx=i)))
    ht = ht.annotate(sim_gt = ht.sim_gt.map(lambda x: hl.struct(GT=x)))
    sim_mt = ht._unlocalize_entries('sim_gt','cols',['col_idx'])
    sim_mt = sim_mt.annotate_globals(recombs = recombs)
    
    return sim_mt

if __name__=='__main__':
    chr_ls = [22]
    ref_mt = get_mt(chr_ls)
    ref_mt = ref_mt.add_row_index('tmp_row_idx')
    ref_mt = ref_mt.filter_rows(ref_mt.tmp_row_idx<200)
    row_ct, Nref= ref_mt.count()
    print(f'... reference dataset: {row_ct} rows, {Nref} cols ...')
    
    Nsim=int(1e7)
    mu=1e-1
    sim_mt = ls_sim(ref_mt=ref_mt, Nsim=Nsim, Nref=Nref, mu=mu)
    print(sim_mt.recomb_ls.show())
    print(len(sim_mt.recomb_ls.collect()[0]))
    
    sim_path = f'gs://nbaya/risk_gradients/ls_sim.Nref_{Nref}.Nsim_{Nsim}.Nsnps_{sim_mt.count_rows()}.mt'
    print(f'... writing to {sim_path} ...')
    sim_mt.write(sim_path)
    
else:
    hl.init(log='/Users/nbaya/Downloads/hail.log')
    ref_mt = hl.methods.balding_nichols_model(2, 100, 200)
    sim_mt = ls_sim(ref_mt=ref_mt, Nsim = 50)
    
    recomb_ls = sim_mt.recombs.collect()[0]
    Nsim=150
    Nref=ref_mt.count_cols()
    sim_mt2 = ls_sim(ref_mt=ref_mt, Nsim=Nsim, recombs=recombs)
    sim_mt2.write(f'/Users/nbaya/Downloads/ls_sim.Nref_{Nref}.Nsim_{Nsim}.Nsnps_{sim_mt.count_rows()}.mt',overwrite=True)
    sim_mt2.describe()
    
    sim_mt_tmp = hl.read_matrix_table(f'/Users/nbaya/Downloads/ls_sim.Nref_{Nref}.Nsim_{Nsim}.Nsnps_{sim_mt.count_rows()}.mt')
    
    import numpy as np
    
#def ls_sim(ref_mt, Nsim, mu=0.01, Nref=None, snps=None):
#    assert (mu > 0) and (mu < 1), 'Mutation rate (recombination rate) must be in (0,1)'
#    assert 'locus' in ref_mt.row, '`locus` row field must be in `ref_mt`'
#
#    Nref = ref_mt.count_cols() if Nref is None else Nref
#    snps = ref_mt['locus'].collect() if snps is None else snps
#    
#    ## Get intervals of non-recombination
#    recomb_idx0 = np.random.binomial(1,mu,size=len(snps)-2) #subtract 2 from length because recombination events can't occur at first or last SNP
#    recomb_idx1 = np.insert(arr=recomb_idx0,obj=0,values=1)
#    recomb_idx2 = np.append(arr=recomb_idx1,values=1)
#    recomb_idx3 = recomb_idx2==1
#    recomb_ls = np.asarray(snps)[recomb_idx3]
#    interval_ls = list(zip(recomb_ls[:-1], recomb_ls[1:]))
#    print(f'number of recombination events: {len(interval_ls)-1}')
#
#    ## get table of intervals
#    it0 = hl.utils.range_table(len(interval_ls))
#    intervals = hl.literal(interval_ls)
#    it0 = it0.key_by(intervals = hl.interval(intervals[it0.idx][0], intervals[it0.idx][1]))
#    
#    ## simulation
#
#    it = it0.annotate(hap = hl.range(Nsim).map(lambda i: [hl.int(hl.rand_unif(0, Nref)), 
#                                                          hl.int(hl.rand_unif(0, Nref))]))
#    ht = ref_mt.localize_entries(entries_array_field_name='entries')
#    ht = ht.annotate(it = it[ht.locus])
#    ht = ht.annotate(sim_gt = ht.it.hap.map(lambda i: hl.parse_call(
#            hl.str(ht.entries[i[0]].GT[0])+'|'+hl.str(ht.entries[i[1]].GT[1]))))
#    ht = ht.annotate_globals(cols = hl.range(Nsim).map(lambda i: hl.struct(col_idx=i)))
#    ht = ht.annotate(sim_gt = ht.sim_gt.map(lambda x: hl.struct(GT=x)))
#    mt_sim = ht._unlocalize_entries('sim_gt','cols',['col_idx'])
#    mt_sim = mt_sim.annotate_globals(recomb_ls = recomb_ls.tolist())
#    return mt_sim


snps = ref_mt.key_rows_by('locus').locus.collect()
recomb_occurs = hl.zip(snps[1:-1], hl.map(lambda x: hl.int(hl.rand_bool(mu)), hl.range(len(snps)-2)))
#        recombs = hl.eval(hl.fold(lambda i, j: i.append(j[0]) if j[1] else i, 
#                                  hl.literal([snps[0]]), recomb_occurs))
        recombs = list(reduce(lambda i, j: i + [j] if hl.eval(hl.rand_bool(mu)) else i, snps[1:-1], [[snps[0]]]))

        