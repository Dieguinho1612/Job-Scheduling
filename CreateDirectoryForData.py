#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import sys


# In[1]:


start = sys.argv[1]
end = sys.argv[2]
DS_numbers = list(range(start,end+1))


# In[4]:


if not os.path.exists('Data'):
    os.mkdir('Data')
for DS in DS_numbers:
    DS_str = "0"*(2-len(str(DS))) + str(DS)
    if not os.path.exists(f'Data/DataSet_{DS_str}'): 
        os.mkdir(f'Data/DataSet_{DS_str}')

