
# coding: utf-8

# In[28]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
#import sigtools
get_ipython().magic(u'matplotlib inline')


# In[ ]:




# In[29]:

d = pd.read_csv('../data_real/Robin-2.csv')
#words = pd.read_csv('../data_real/words_zhanna_labeled.csv')



# In[30]:

lines = open('../data_real/eeg_robin_channels.txt').readlines()
lines = lines[1:]
lines = [line.strip() for line in lines]

CHANNELS = dict()
for line in lines:
    num, _, label = line.partition(' ')
    num = int(num) - 1
    CHANNELS[num] = label


# In[31]:

lines


# In[32]:

d.shape


# In[33]:

d.head()


# In[34]:

plt.plot(d.channel_0)
plt.plot(d.channel_1)
plt.plot(d.channel_2)
plt.plot(d.channel_3)


# In[35]:

_ = plt.specgram(d.channel_2, NFFT=128, Fs=250, noverlap=120)#what is our sampling frequency with EMOTIV ?


# In[36]:

eeg = np.array(d.ix[:, 0:14])
eeg = eeg[250:15000,]


# In[ ]:




# In[37]:

from scipy import signal


# In[38]:

# filter from 5 to 35 Hz, helps remove 60Hz noise (that we dont have here, since it works on battery) and replicates paper
## also helps remove the DC line noise (baseline drift)
## 125 is half the sampling rate (250Hz/2)
b, a = signal.butter(4, (10.0/125, 35.0/125), btype='bandpass') 
b, a


# In[39]:

eeg_f = signal.lfilter(b, a, eeg, axis=0)
#eeg_f.shape
eeg.shape


# In[40]:

## the filter needs a couple samples to converge
## honestly 500 is very conservative, but we don't these samples anyway so whatever

eeg_f = eeg_f[500:]


# In[41]:

plt.figure(figsize=(14, 4))
plt.plot(eeg[500:15000, 13]) ## raw data


# In[42]:

plt.figure(figsize=(14, 4))
plt.plot(eeg_f[500:15000, 13])


# In[43]:

from sklearn.decomposition import FastICA


# In[44]:

ica = FastICA()
sources = ica.fit_transform(eeg_f)
means = ica.mean_.copy()
mixing = ica.mixing_.copy()


# In[51]:

## look at the plots of each channel to find the eyeblink component
## TODO: make a more robust eyeblink component finder
for i in range(ica.components_.shape[0]):
    plt.figure()
    plt.plot(sources[500:15000, i])
    plt.title(i)


# In[52]:

eye_blinks_ix = 10


# In[53]:

mixing[:, eye_blinks_ix] = 0 # setting eyeblink component to 0
eeg_ff = sources.dot(mixing.T) + means # this is the ICA inverse transform



# In[54]:

plt.figure()
plt.plot(eeg_f[:15000, 13])

plt.figure()
plt.plot(eeg_ff[:15000, 13])



# In[55]:




# In[56]:




# In[25]:




# In[26]:




# In[27]:




# In[ ]:



