#!/usr/bin/env python
# coding: utf-8

# In[1]:


from commstools import Signal
from commstools.impairments import apply_awgn, apply_pmd
import numpy as np


# In[2]:


NUM_SYMBOLS = 2**18
SYMBOL_RATE = 1e9
SPS = 2
MOD = "PSK"
ORDER = 4
ESN0_DB = 20
DGD_SYMBOLS = 0.0
THETA = np.pi / 4


# In[3]:


sig = Signal.psk(
    num_symbols=NUM_SYMBOLS,
    sps=SPS,
    symbol_rate=SYMBOL_RATE,
    order=ORDER,
    num_streams=2,
    seed=42,
)
sig.print_info()
sig.plot_waveform(num_symbols=100, show=True)
sig.plot_psd(show=True, nperseg=2**10)
sig.plot_constellation(show=True)


# In[4]:


sig_dist = sig.copy()

sig_dist.print_info()
sig_dist = apply_awgn(sig_dist, esn0_db=ESN0_DB)
sig_dist.plot_waveform(num_symbols=100, show=True)
sig_dist.plot_psd(show=True, nperseg=2**10)
sig_dist.plot_constellation(show=True)


# In[5]:


sig_dist = apply_pmd(sig_dist, dgd=DGD_SYMBOLS / SYMBOL_RATE, theta=THETA)

sig_dist.print_info()
sig_dist.plot_waveform(num_symbols=100, show=True)
sig_dist.plot_psd(show=True, nperseg=2**10)
sig_dist.plot_constellation(show=True)


# In[23]:


sig_dist.matched_filter()

sig_dist.print_info()
sig_dist.plot_waveform(num_symbols=100, show=True)
sig_dist.plot_psd(show=True, nperseg=2**10)
sig_dist.plot_constellation(show=True)


# In[38]:


sig_to_eq = sig_dist.copy()

sig_to_eq.equalize(
    method="rls", num_train_symbols=2**10, block_size=1, num_taps=11, device="cpu"
)


# In[39]:


sig_to_eq.plot_equalizer(show=True)


# In[40]:


sig_to_eq.print_info()
sig_to_eq.plot_waveform(num_symbols=100, show=True)
sig_to_eq.plot_psd(show=True, nperseg=2**10)
sig_to_eq.plot_constellation(show=True)


# In[41]:


sig_to_eq.resolve_symbols()
sig_to_eq.evm()


# In[11]:


sig_to_eq.resolved_symbols


# In[12]:


sig_to_eq.source_symbols
