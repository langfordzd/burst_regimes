#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 15:20:44 2022

@author: zachary
"""
#%%
def pow_chars(electrode):
    
    import numpy as np
    import config
    from scipy import ndimage

    loc, chan, ep, raw, amps, freqs, phas, tfr, tfr_st_means,b,ids = zip(*electrode)
    tfr = np.array(tfr)   
    t_num = len(tfr_st_means)
    min_cycs = 2
    threshold = 6    
    recov = []
    count = np.zeros((t_num))   
    
    for epoch in range(t_num):
        
        tDat = tfr[epoch,:,:]
        ars = tDat > threshold
        labeled_image, num_features = ndimage.label(ars)
        objs = ndimage.find_objects(labeled_image)
        burstInds = []
        
        for ob in objs:
            burstInds.append([ob[1].start,ob[1].stop, int((ob[0].stop + ob[0].start)/2)])      
                
        for whichInd, ind in enumerate(burstInds):
            if ind[0] in config.keeps-10 and ind[1] in config.keeps+10:
                duration = ind[1]-ind[0]
                if duration > (config.sfreq/config.freqs[ind[2]] * min_cycs):
                    start = ind[0]
                    end = ind[1]
                    duration = (ind[1] - ind[0]) / config.sfreq
                    recov.append([loc[epoch],chan[epoch],ep[epoch],ids[epoch],start,end,duration])
                    count[epoch] =  count[epoch]+1
    return recov
#%%
def cycler_worker(t, toE):
    from bycycle.features import compute_features
    import numpy as np
    import config
    import warnings
    warnings.filterwarnings("ignore")
    from scipy import stats
    loc, chan, ep, e, a, f, p, tf, ts, b, ids = config.trials[t]       
    toEvaluate = config.whichToEval[toE]
    signalFeats = compute_features(e, config.sfreq, 
                              config.f_beta, threshold_kwargs={'amp_fraction_threshold': toEvaluate[0], 
                                                            'amp_consistency_threshold': toEvaluate[1],
                                                            'period_consistency_threshold': toEvaluate[2],
                                                            'monotonicity_threshold': toEvaluate[3],
                                                            'min_n_cycles': toEvaluate[4]})
    bursts = signalFeats[signalFeats['is_burst']]   
    pred = np.zeros(e.shape)
    recov = list()
    count = 0
    if any(bursts['is_burst']):
        burst_list = np.split(bursts, np.flatnonzero(np.diff(bursts.index) != 1) + 1)
        for num, whichBurst in enumerate(burst_list):
            start = min(whichBurst['sample_last_trough'])
            end = max(whichBurst['sample_next_trough'])              
            if start in config.keeps-10 and end in config.keeps+10:
                recov.append([loc, chan, ep, ids, start, end, (end-start)/config.sfreq])
                pred[start:end] = 1
                count = count+1
                
    stat        = stats.spearmanr(pred[config.keeps],a[config.keeps]).correlation
    toSend      = config.bop_recov[config.bop_recov['ids'] == ids].values.tolist()   
    boc_u,boc_c   = out_and_in(recov,toSend)
    bop_u,bop_c   = out_and_in(toSend,recov)
    bop_uniq     = collect_chars_split(bop_u).dropna()
    boc_uniq     = collect_chars_split(boc_u).dropna() 
    boc_co       = collect_chars_split(boc_c).dropna()
    bop_co       = collect_chars_split(bop_c).dropna()    
    return  loc, chan, ep, ids, t, toE, stat, count, boc_uniq, boc_co, bop_uniq, bop_co
#%%
def out_and_in(outside,inside):
    share_percent = 0.5
    import numpy as np
    o_uniq=[]
    com=[]
    a=[]
    b=[]
    for o_ind,o_burst in enumerate(outside):
        uniq = True
        o_times = np.arange(o_burst[4],o_burst[5])
        for i_ind,i_burst in enumerate(inside):
            if o_burst[3] == i_burst[3]:
                i_times = np.arange(i_burst[4],i_burst[5])
                o_oi = len(np.intersect1d(o_times,i_times)) / len(o_times)
                o_io = len(np.intersect1d(o_times,i_times)) / len(i_times)
                if (o_oi>=share_percent) and (o_io>=share_percent):
                    uniq = False
                    com.append(o_burst) 
                    b.append(o_ind)
        if uniq == True:
            a.append(o_ind)
            o_uniq.append(o_burst)
    return o_uniq,com
 #%%
def collect_chars_split(which):
    import numpy as np
    import pandas as pd
    import config
    m = []
    for whichBurst in which:
        sess = config.trials[whichBurst[3]]
        locations = np.arange(whichBurst[4], whichBurst[5])
        freq = sess[5][locations]
        freq[freq == 0] = np.nan
        amp = sess[4][locations]
        amp[amp == 0] = np.nan
        m.append([whichBurst[0], whichBurst[1], whichBurst[2] ,whichBurst[3], whichBurst[4], whichBurst[5], np.nanmean(amp), np.nanmean(freq),whichBurst[6]]) 
    mpd = pd.DataFrame(m, columns=['loc', 'chan', 'ep','ids', 'start', 'end', 'amp','freq','duration'])       
    mpd = mpd.dropna().drop_duplicates()    
    return mpd

