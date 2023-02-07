#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Calculate the spatial depth of vector v relative
# to all columns of x.
def sdepth(v, x):
    p, n = x.shape
    z = x - v[:, None]
    zn = np.sqrt((z**2).sum(0))
    zn[np.abs(zn) < 1e-12] = np.inf
    z /= zn
    u = z.mean(1)
    return 1 - np.sqrt(np.sum(u**2))

# Calculate the L2 depth of vector v relative
# to all columns of x.
def l2depth(v, x):
    p, n = x.shape
    z = x - v[:, None]
    zn = np.sqrt((z**2).sum(0))
    d = zn.mean()
    return 1e6 / (1 + d)


# In[50]:


from numpy.random import seed
from numpy.random import normal
import numpy as np
import pandas as pd

seed(1)

x = np.random.normal(size=(1600, 30))
epsilon = np.random.normal(size=(1600, 30)) # the larger the variance goes, the less correlated x and y is
b = np.random.normal(size=(1600, 30))
y = np.multiply(x, b)+epsilon
# depth = sdepth(x[0,:], x.T)


# In[10]:


# get github site, get literature writing beginning. 
# data depth, data depth association, 
# what are standard method of association

# overleaf, github

# data depth with correlation
# depth is one x with respect to all x's, write a loop to get through all of the x 
# sort x (concat y) with respect to depth desc, get top 10%, 20%, etc. (deepest points)
# after those densed points are the central region. 
# then use the ellipsoid volumn. 
# then do the formula


# In[55]:


# get depth
depth = []
for xi in x:
    xi_reshaped = np.array([xi]).T
    depth.append(l2depth(xi_reshaped, x))
len(depth)


# In[58]:


# pair depth with x 

df = pd.DataFrame(columns=('x', 'depth'))
for i in range(len(depth)):
    df.loc[i] = [x[i], depth[i]]
print(df)


# In[39]:


"""
# Procedure: constructing a two-way dictionary
# Purpose: sort array_a wrt array_b
# reference: https://stackoverflow.com/questions/1456373/two-way-reverse-map
# Usage: 

>>> d = TwoWayDict()
>>> d['foo'] = 'bar'
>>> d['foo']
'bar'
>>> d['bar']
'foo'
>>> len(d)
1
>>> del d['foo']
>>> d['bar']
Traceback (most recent call last):
  File "<stdin>", line 7, in <module>
KeyError: 'bar'

"""

class TwoWayDict(dict):
    def __setitem__(self, key, value):
        # Remove any previous connections with these values
        if key in self:
            del self[key]
        if value in self:
            del self[value]
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)

    def __len__(self):
        """Returns the number of connections"""
        return dict.__len__(self) // 2


# In[37]:


# sort depth desc
sort_desc = np.sort(depth)[::-1]
print(sort_desc)

# concat y


# In[ ]:


# get volumn


# In[7]:


# sort desc
sorted = np.sort(corr)[::-1]
max_depth = sorted[0]
n = len(sorted)
volumns = []

# get volumn, step = 10%
step = int(0.1*n)
for k in range(0, 3201, step):
    alpha = k/n
    vol = np.corrcoef(depth[0:k])
    print("Vol:", vol)
    volumns.append(vol)
volumns


# In[ ]:




