#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 09:19:12 2021

@author: zachary
"""

import numpy as np
import pandas as pd
r_seed = 0
n_cycles = 7
t_num = 100
sfreq = 500
trial_len = 1
trial_points = sfreq*trial_len
t_between = 0.5
iti_points = int(sfreq*t_between) #one on each side so iti is actually double this
tot_points = trial_points + 2*iti_points
filter_edge = 35
freqs = np.arange(15, 29.1, 0.5)
f_beta = (15,29.)
shared_time_points = 0.5
#thresholds = np.linspace(0.1,16)
thresholds = [6]# np.linspace(5.9,6.1)
keeps = np.arange(iti_points,tot_points-iti_points)
cvfolds = 10
#cvfolds = t_num
sims = list()

pt_recov = pd.DataFrame()
trials = list()
whichToEval = list()
n_cycles = 7