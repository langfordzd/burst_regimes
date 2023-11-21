#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 14:22:51 2021

@author: zachary
"""
#%%
def simmer(r_seed,t_num,depth_point,cycles,distribution,probs):
   
    import numpy as np
    import random
    from neurodsp.filt import filter_signal
    from neurodsp.sim import sim_powerlaw, sim_cycle
    from scipy.stats import uniform, lognorm
    from mne.time_frequency import tfr_array_morlet
    from scipy import signal as sp_sig

    import pandas as pd
    import config as cfg
    from neurodsp.timefrequency import amp_by_time, freq_by_time
    #r_seed, t_num, em, depth_point, cycles, distribution, probs, names = packet
    random.seed(r_seed)
    np.random.seed(r_seed)

    sfreq = cfg.sfreq
    trial_points = cfg.trial_points
    iti_points = cfg.iti_points
    trial_and_iti_points = trial_points+2*iti_points
    trial_chars = []
    sig_length = trial_and_iti_points * cfg.t_num
    sig_length_sec = sig_length/sfreq
    bot = np.arange(iti_points, sig_length, trial_points+2*iti_points)
    top = bot+cfg.trial_points
    signal = np.zeros(sig_length)
    sig_on_off = np.zeros(sig_length)

    population = [False,True]
    burst_last = False
    current_point = 0
    amps_num = 5000
    if distribution[0] == 'uniform':
        amp_dist = uniform.rvs(loc=distribution[1],scale=distribution[2],size=amps_num)
        amp_dist = amp_dist[:2500]
    elif distribution[0] == 'lognorm':
        #outlier  = distribution[4]
        amp_dist = lognorm.rvs(distribution[1],loc=distribution[2],scale=distribution[3],size=amps_num,random_state=r_seed)
        #amp_dist = amp_dist[amp_dist<distribution[2]+outlier*distribution[3]]
        amp_dist = amp_dist[:2500]


    from scipy.stats import truncnorm
    
    def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
        return truncnorm(
            (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
    X = get_truncated_normal(mean=21, sd=1, low=15, upp=29)

    if not distribution[0] == 'noise':
        while current_point < sig_length-cfg.iti_points: 
            if random.choices(population,probs)[0] and not burst_last:
                cycs = np.random.uniform(cycles[0], cycles[1]) 
                #cyc_freq = np.random.uniform(18, 25) 
                cyc_freq = float(X.rvs(1))
                n_sec = cycs/cyc_freq
                n_sam = int(np.ceil(n_sec*sfreq))
                cyc_s = (sfreq / cyc_freq) / sfreq
                osc_amp =  np.random.choice(amp_dist)  
                rdsym = np.random.uniform(0.25,0.75)
                cyc = sim_cycle(cyc_s, sfreq, cycle_type='asine', rdsym = rdsym)
                burst = osc_amp * np.tile(cyc, int(np.ceil(cycs)))[:n_sam]
                modpoint = n_sec/np.random.uniform(1,3)
                dp = np.random.uniform(depth_point[0], depth_point[1]) #np.random.uniform(osc_amp + depth_point[0], osc_amp + depth_point[1]) 
                depth = (1/osc_amp) / dp
                # depth = (osc_amp) / dp
                # if osc_amp < 0.25:
                #     depth = osc_amp/ 1

                samples = np.arange(burst.size) / sfreq
                mod_amp = np.exp(-(samples - modpoint)**2 / depth)
                amp_mod_signal = burst*mod_amp
                in_bounds = np.arange(current_point,amp_mod_signal.size + current_point)
                #plt.plot(amp_mod_signal)
                
                for trial, (a,b) in enumerate(zip(bot,top)): 
                    if np.all((in_bounds > a) & (b > in_bounds)): 
                        trial_chars.append([current_point, current_point + burst.size, 
                                            current_point-a+cfg.iti_points ,current_point+burst.size-a+cfg.iti_points,
                                            cyc_freq, cycs, n_sec, osc_amp, depth, rdsym,trial])
                        signal[in_bounds] = amp_mod_signal
                        sig_on_off[in_bounds] = 1
                        burst_last = True  
                        current_point = current_point+amp_mod_signal.size
                        # bot = np.delete(bot,np.arange(0,trial-1))
                        # top = np.delete(top,np.arange(0,trial-1))
                        break
        
            else:
                burst_last = False
                cycs = np.random.uniform(1,9)
                cyc_freq = np.random.uniform(15,29)
                current_point = current_point+int((cycs/cyc_freq) * sfreq)
                
    pl = sim_powerlaw(sig_length_sec, sfreq, exponent=-2)  
    signal = signal[0:sig_length]    
    signal_noise = np.sum([signal, pl], axis=0)
    bot = np.arange(iti_points, sig_length, trial_points+2*iti_points)
    top = bot+cfg.trial_points
    inter_ind = pd.IntervalIndex.from_arrays(bot, top, closed='both')
    if distribution[0] == 'noise':
        chars = pd.DataFrame(np.nan,index=[0], columns=['start', 'end', 'trial_start','trial_end','freq', 'n_cycs', 
                                                        'n_secs', 'amp', 'depth', 'rdsym','trial_number','trial'])
    else:
        chars = pd.DataFrame(trial_chars) 
        chars.columns =['start', 'end','trial_start','trial_end', 'freq', 'n_cycs', 'n_secs', \
                    'amp', 'depth', 'rdsym','trial_number']    
        chars['trial'] = pd.cut(chars['start'],bins=inter_ind)   
    #print(len(chars))
    #print(chars['n_secs'][0])
    es = np.reshape(signal_noise, (t_num, 1, trial_and_iti_points))
    tfr = np.squeeze(tfr_array_morlet(es, cfg.sfreq, cfg.freqs, n_cycles=cfg.n_cycles, 
                                      output='power', use_fft=False, 
                                      zero_mean=True, n_jobs=10, 
                                      verbose=False))
    for f in range(0,tfr.shape[1]):
        freq = np.squeeze(tfr[:,f,:])
        med = np.median(freq[:,cfg.keeps], axis=(0,1))
        print([f,6*med])
        tfr[:,f,:] =  freq / med   

    f_range = (None, 45) 
    sig = filter_signal(signal_noise, cfg.sfreq, 'lowpass', f_range, remove_edges=False)
    trials = np.reshape(sig, (t_num, trial_and_iti_points))
    trials_centered = np.array([2.*(t - np.min(t))/np.ptp(t)-1 for t in trials])
        
    amps = np.nan_to_num(amp_by_time(signal_noise, cfg.sfreq, cfg.f_beta, 
           #hilbert_increase_n = False,
           remove_edges=False))
    
    freqs = sp_sig.medfilt(np.nan_to_num(freq_by_time(signal_noise, cfg.sfreq, cfg.f_beta,
                         remove_edges=False,#hilbert_increase_n = False
                         )))
    
    amps = np.reshape(amps, (t_num, trial_and_iti_points))
    amps_centered = np.array([1.*(t - np.min(t))/np.ptp(t)-0 for t in amps])
    freqs = np.reshape(freqs, (t_num, trial_and_iti_points))

    listed = []
   #[loc,chan,ep,e,a,f,p,tf,ts,b,ids]
    inter_ind = inter_ind.to_tuples()
    for t in range(t_num):
        listed.append(['sim',
                       0,
                       t,
                       trials_centered[t,:],
                       amps_centered[t,:],
                       freqs[t,:],
                       trials_centered[t,:],
                       tfr[t,:,:],
                       1,
                       list(inter_ind[t]),
                       t])
    
    return listed, chars