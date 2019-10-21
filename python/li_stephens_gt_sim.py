#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 17:12:06 2019

Li-Stephens genotype simulation

@author: nbaya
"""



import hail as hl
from hail.typecheck import typecheck, oneof, nullable
from hail.matrixtable import MatrixTable


def get_mt(chr_ls):
    variants = hl.import_table('gs://nbaya/split/hapmap3_variants.tsv')
    variants = variants.annotate(**hl.parse_variant(variants.v))
    variants = variants.key_by('locus','alleles') 
    gt0 = hl.import_bgen(path='gs://fc-7d5088b4-7673-45b5-95c2-17ae00a04183/imputed/ukb_imp_chr'+str(set(chr_ls)).replace(' ','')+'_v3.bgen',
                         entry_fields=['GT'],
                         n_partitions = 1000,
                         sample_file = 'gs://ukb31063/ukb31063.autosomes.sample',
                         variants=variants)
    ht0 = hl.read_matrix_table(f'gs://nbaya/split/ukb31063.hm3_variants.gwas_samples_repart.mt').cols()
    #Remove withdrawn samples
    withdrawn = hl.import_table('gs://nbaya/w31063_20181016.csv',missing='',no_header=True)
    withdrawn = withdrawn.key_by('f0')
    ht0 = ht0.filter(hl.is_defined(withdrawn[ht0.s]),keep=False)
    
    gt0 = gt0.filter_cols(hl.is_defined(ht0[gt0.s]))
    
    return gt0

@typecheck(ref_mt=MatrixTable,
           Nsim=int,
           mu=nullable(oneof(float,
                             int)),
           recomb_snps=nullable(list),
           seed=nullable(int))
def li_stephens_sim(ref_mt, Nsim, mu=0.01, recomb_snps=None, seed=None):
    r'''
    Simulates `Nsim` new individuals by randomly copying and pasting chunks of 
    genotypes from individuals in `ref_mt`, according to the Li-Stephens model.
    Points of recombination are defined by `recomb_snps`.
    
    Parameters
    ----------
    ref_mt : :class:`.MatrixTable`
        :class:`.MatrixTable` containing "reference" genotypes for simulation.
        `ref_mt` must have a locus row field named `locus` and a phased 
        CallExpression entry field named `GT`.
    Nsim : :obj:`int`
        Number of individuals to simulate.
    mu : :obj:`float` or :obj:`int`, optional
        Mutation rate (or recombination rate) for creating recombination events.
    recomb_snps : :obj:`list`, optional
        List of SNPs where recombination events occur, including the first and
        last SNP of the SNP range where recombinations will occur.
    seed : :obj:`int`, optional
        Random seed for generation of recombination events.
        
    Returns
    -------
    :class: `.MatrixTable`
        :class:`.MatrixTable` with simulated genotypes as the entry field `GT`.
        `recomb_snps` are also annotated as a global field. If not `None`, `mu`
        and `seed` are also annotate as global fields.
    '''
    if recomb_snps is None:
        assert (mu >= 0) and (mu <= 1), 'Mutation rate (recombination rate) must be in [0,1]'
    else:
        print('Ignoring mutation rate `mu` because `recomb_snps` is not None')
    assert 'locus' in ref_mt.row, '`locus` row field must be in `ref_mt`'

    ## get table of recombination hotspots
    it0 = ref_mt.rows().key_by('locus')    
    it0 = it0.add_index('tmp_idx')
    Nsnps = it0.count() 
    if recomb_snps is None:
        it0 = it0.annotate(recomb=(hl.rand_bool(mu,seed=seed))|(it0.tmp_idx==0)|(it0.tmp_idx==Nsnps-1))
        recomb_snps = it0.filter(it0.recomb==1).locus.collect()
    intervals = list(map(lambda start, stop: hl.interval(start, stop), recomb_snps[:-1], recomb_snps[1:]))
    intervals = intervals[:-1] + [hl.interval(intervals[-2].end, intervals[-1].end, includes_end=True)]
    it0 = hl.Table.parallelize([hl.struct(interval=interval) for interval in intervals])
    it0 = it0.key_by('interval')
    
    ## simulation
    Nref = ref_mt.count_cols()
    it = it0.annotate(hap = hl.range(Nsim).map(lambda i: [hl.int(hl.rand_unif(0, Nref)), 
                                                          hl.int(hl.rand_unif(0, Nref))]))
    ht = ref_mt.localize_entries(entries_array_field_name='entries')
    ht = ht.annotate(it = it[ht.locus])
    ht = ht.annotate(sim_gt = ht.it.hap.map(lambda i: hl.parse_call(
            hl.str(ht.entries[i[0]].GT[0])+'|'+hl.str(ht.entries[i[1]].GT[1]))))
    ht = ht.annotate_globals(cols = hl.range(Nsim).map(lambda i: hl.struct(col_idx=i)))
    ht = ht.annotate(sim_gt = ht.sim_gt.map(lambda x: hl.struct(GT=x)))
    sim_mt = ht._unlocalize_entries('sim_gt','cols',['col_idx'])
    sim_mt = sim_mt.annotate_globals(
            ls_sim = hl.struct(mu='' if mu is None else mu,
                               recomb_snps=recomb_snps,
                               seed='' if seed is None else seed))    
    sim_mt = sim_mt.drop('entries')
    return sim_mt

if __name__=='__main__':
    chr_ls = list(range(1,23))
    ref_mt = get_mt(chr_ls)
#    ref_mt = ref_mt.add_row_index('tmp_row_idx')
#    ref_mt = ref_mt.filter_rows(ref_mt.tmp_row_idx<200)
    row_ct, col_ct= ref_mt.count()
    print(f'\n\n... reference dataset: {row_ct} rows, {col_ct} cols ...\n')
    
    Nsim=int(100)
    mu=1e-2
    sim_mt = li_stephens_sim(ref_mt=ref_mt, Nsim=Nsim, mu=mu)
#    print(sim_mt.ls_sim.recomb_snps.show())
#    print(len(sim_mt.ls_sim.recomb_snps.collect()[0]))
    
    sim_path = f'gs://nbaya/risk_gradients/ls_sim.Nref_{col_ct}.Nsim_{Nsim}.Nsnps_{sim_mt.count_rows()}.mt'
    print(f'\n\n... writing to {sim_path} ...\n')
    sim_mt.write(sim_path,overwrite=True)
    
#else:
#    hl.init(log='/Users/nbaya/Downloads/hail.log')
#    ref_mt = hl.methods.balding_nichols_model(2, 100, 200)
#    sim_mt = ls_sim(ref_mt=ref_mt, Nsim = 50)
#    
#    recomb_snps = sim_mt.recomb_snps.collect()[0]
#    Nsim=150
#    Nref=ref_mt.count_cols()
#    sim_mt2 = ls_sim(ref_mt=ref_mt, Nsim=Nsim, recomb_snps=recomb_snps)
#    sim_mt2.write(f'/Users/nbaya/Downloads/ls_sim.Nref_{Nref}.Nsim_{Nsim}.Nsnps_{sim_mt.count_rows()}.mt',overwrite=True)
#    sim_mt2.describe()
#    
#    sim_mt_tmp = hl.read_matrix_table(f'/Users/nbaya/Downloads/ls_sim.Nref_{Nref}.Nsim_{Nsim}.Nsnps_{sim_mt.count_rows()}.mt')
    
    
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
#
#
#snps = ref_mt.key_rows_by('locus').locus.collect()
#recomb_occurs = hl.zip(snps[1:-1], hl.map(lambda x: hl.int(hl.rand_bool(mu)), hl.range(len(snps)-2)))
##        recombs = hl.eval(hl.fold(lambda i, j: i.append(j[0]) if j[1] else i, 
##                                  hl.literal([snps[0]]), recomb_occurs))
#        recombs = list(reduce(lambda i, j: i + [j] if hl.eval(hl.rand_bool(mu)) else i, snps[1:-1], [[snps[0]]]))

        