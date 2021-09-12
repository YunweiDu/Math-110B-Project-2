#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


# In[2]:


beta=np.random.uniform(1,2,50)
y=50
gamma=0.95*min(1/beta)
# IGD, the ordering is permitted to have replacement. 


# In[3]:


# IGD, the ordering is permitted to have replacement. 
def IGD_wr_task2(beta,y):
    n = len(beta)
    ordering = np.random.choice(n, n, replace=True)    
    # implement the algorithm's iteration of IGD. Your result should return the the final xk
    # at the last iteration and also the history of objective function at each xk.
    x0=0
    x=[x0]
    z=[]
    for k in range(n):
        beta_k=beta[ordering[k]]
        x1=x0-gamma*beta_k*(x0-y)
        x0=x1
        z.append(1/2*sum(beta*(x1-y)**2))
        x.append(x1)
    return z,x,x1
# implement the algorithm's iteration of IGD. Your result should return the the final xk
# at the last iteration and also the history of objective function at each xk.


# In[4]:


# IGD, the ordering is not permitted to have replacement.
def IGD_wo_task2(beta,y):
    n = len(beta)
    ordering = np.random.choice(n, n, replace=False)
    # implement the algorithm's iteration of IGD. Your result should return the the final xk
    # at the last iteration and also the history of objective function at each xk.
    x0=0
    z=[]
    x=[x0]
    for k in range(n):
        beta_k=beta[ordering[k]]
        x1=x0-gamma*beta_k*(x0-y)
        x0=x1
        z.append(1/2*sum(beta*(x1-y)**2))
        x.append(x1)
    return z,x,x1


# In[5]:


z_1,xk_1,x_1=IGD_wr_task2(beta,y)

z_2,xk_2,x_2=IGD_wo_task2(beta,y)


# In[6]:


##plot the history##
for history in [z_1,z_2]:
    plt.plot(history)


# In[ ]:




