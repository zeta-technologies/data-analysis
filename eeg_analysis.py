
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().magic(u'matplotlib inline')

d_prime= pd.read_csv('file')
d = pd.read_csv('file')



# In[45]:

lines = open('../data_real/eeg_zhanna_channels.txt').readlines()
lines = lines[1:]
lines = [line.strip() for line in lines]

CHANNELS = dict()
for line in lines:
    num, _, label = line.partition(' ')
    num = int(num) - 1
    CHANNELS[num] = label


# In[46]:

d.shape


# In[47]:

d.head()


# In[48]:

plt.plot(d.channel_0)


# In[57]:

_ = plt.specgram(d.channel_1, NFFT=128, Fs=250, noverlap=64)


# In[50]:

eeg = np.array(d.ix[:, 0:7])
eeg = eeg[250:7750,]

tag = np.array(d.ix[:, 'tag'])
tag = tag[250:]


# In[51]:

from scipy import signal


# In[52]:

# filter from 5 to 35 Hz, helps remove 60Hz noise and replicates paper
## also helps remove the DC line noise (baseline drift)
## 125 is half the sampling rate (250Hz/2)
b, a = signal.butter(4, (2.0/125, 35.0/125), btype='bandpass') 
b, a


# In[53]:

eeg_f = signal.lfilter(b, a, eeg, axis=0)


# In[54]:

## the filter needs a couple samples to converge
## honestly 500 is very conservative, but we don't these samples anyway so whatever
eeg_f = eeg_f[500:]
#tag = tag[500:]


# In[61]:

plt.figure(figsize=(14, 4))
plt.plot(eeg[:, 1]) ## raw data
b2, a2 = signal.butter(4, (2.0/125, 35.0/125), btype='bandpass') 
b2, a2


# In[60]:

_ = plt.specgram(eeg[:,1], NFFT=128, Fs=250, noverlap=64)


# In[56]:

plt.figure(figsize=(14, 4))
plt.plot(eeg_f[300:9000, 1])


# In[40]:

from sklearn.decomposition import FastICA


# In[41]:

ica = FastICA()
sources = ica.fit_transform(eeg_f)
means = ica.mean_.copy()
mixing = ica.mixing_.copy()


# In[42]:

## look at the plots to find the eyeblink component
## TODO: make a more robust eyeblink component finder
for i in range(ica.components_.shape[0]):
    plt.figure()
    plt.plot(sources[:8500, i])
    plt.title(i)
    


# In[43]:

eye_blinks_ix = 7


# In[19]:

mixing[:, eye_blinks_ix] = 0 # setting eyeblink component to 0
eeg_ff = sources.dot(mixing.T) + means # this is the ICA inverse transform


# In[20]:

plt.figure()
plt.plot(eeg_f[500:9000, 7])

plt.figure()
plt.plot(eeg_ff[:8500, 7])


# In[21]:

word_starts = []
prev_t = None

for i, t in enumerate(tag):
    if t != 'focus' and t != '0' and t != prev_t:
        w = word_dict[t]
        word_starts.append( {'index': i, 
                             'word': t,
                             'dict': w} )
    prev_t = t


# In[ ]:

## this confirms that there's ~2.5 seconds between words
np.diff([x['index'] for x in word_starts]) / 250.0


# In[ ]:

recognized = np.array([w['dict']['recognized'] for w in word_starts])


# In[ ]:

eeg_trials = np.zeros((8, len(word_starts), int(250*2.5)))
time = np.arange(0, eeg_trials.shape[2], 1) / 250.0 - 0.5


# In[ ]:

for c in range(8):
    for i in range(len(word_starts)):
        d = word_starts[i]
        start = d['index']
        # 125 samples = 0.5s, 500 samples = 2.0 s
        # we want 0.5s before the stimulus presentation and 2.0 seconds after
        eeg_trials[c, i, :] = eeg_ff[start-125:start+500, c] 


# In[ ]:

# for c in range(8):
#     plt.figure(figsize=(14, 4))
#     _ = plt.plot(time, np.mean(eeg_trials[c], axis=0))
#    plt.title(CHANNELS[c])


# In[ ]:

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# In[ ]:

## this shows the ERPs for -0.5s to 2.5s with 0s = when stimulus is shown
## blue is for remembered words
## red is for not remembered words

N_AVG = 10

for i in range(8):
    plt.figure(figsize=(14, 4))
    
    rec = np.mean(eeg_trials[i][recognized], axis=0)
    rec = moving_average(rec, n=N_AVG)
    _ = plt.plot(time[(N_AVG-1):], rec, c='blue')
    
    not_rec = np.mean(eeg_trials[i][~recognized], axis=0)
    not_rec = moving_average(not_rec, n=N_AVG)
    _ = plt.plot(time[(N_AVG-1):], not_rec, c='red')
    plt.title(CHANNELS[i])
    
    plt.xlabel('Time since stimulus (s)')
    plt.ylabel('EEG amplitude (arbitrary units)')


# ## 

# In[ ]:



