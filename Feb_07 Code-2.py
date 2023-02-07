#!/usr/bin/env python
# coding: utf-8

# In[70]:


# Done: set up github site, overleaf
# In Progress: what are standard method of association

# Done: 
# data depth with correlation
# depth is one x with respect to all x's, write a loop to get through all of the x 
# sort x (concat y) with respect to depth desc, get top 10%, 20%, etc. (deepest points)
# after those densed points are the central region. 
# then use the ellipsoid volumn. 
# then do the formula


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


# In[55]:


# get depth
depth = []
for xi in x:
    xi_reshaped = np.array([xi]).T
    depth.append(l2depth(xi_reshaped, x))
len(depth)


# In[59]:


# pair depth with x 
df = pd.DataFrame(columns=('x', 'y', 'depth'))
for i in range(len(depth)):
    df.loc[i] = [x[i], y[i], depth[i]]
print(df)


# In[64]:


# sort, get central region
df = df.sort_values(by=['depth'], ascending=False)

_10pect = df[:160]
_20pect = df[:160*2]
_30pect = df[:160*3]
print(_30pect)


# #### Get stucked here

# In[68]:


# get volumn
# _10pect.shape()
# print(np.linalg.det(_10pect))


# In[ ]:




