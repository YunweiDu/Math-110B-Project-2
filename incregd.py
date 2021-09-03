# generate a vector of random numbers which obeys the given distribution.
#
# n: length of the vector
# mu: mean value
# sigma: standard deviation.
# dist: choices for the distribution, you need to implement at least normal
#       distribution and uniform distribution.
#
# For normal distribution, you can use ``numpy.random.normal`` to generate.
# For uniform distribution, the interval to sample will be [mu - sigma/sqrt(3), mu + sigma/sqrt(3)].
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def generate_random_numbers(n, mu, sigma, dist="normal"):
    mu=mu
    sigma=sigma
    if dist == "normal":
        return np.random.normal(mu, sigma, n)
    elif dist == "uniform":
        return np.random.uniform(mu - sigma/math.sqrt(3), mu + sigma/math.sqrt(3),n)
    else:
        raise Exception("The distribution {unknown_dist} is not implemented".format(unknown_dist=dist))


# test your code:
y_test = generate_random_numbers(5, 0, 0.1, "normal")
#print(y_test)




y1 = generate_random_numbers(10, 0.5, 1.0, "normal")
y2 = generate_random_numbers(10, 0.5, 1.0, "uniform")
#print(y2)

# IGD, the ordering is permitted to have replacement.
def IGD_wr_task1(y):
    n = len(y)
    gamma=np.zeros(n)
    x=np.zeros(n)
    z = np.zeros (len(x))
    ordering = np.random.choice(n, n, replace=True)

    for k in range(len(y)):
        for ordering in range(len(y)):
            gamma[k] = 1/(k+1)
            x[k]=x[k]-gamma[k]*(x[k]-y[ordering])
            z[k]=(1/2)*sum([(x[k]-y[i])**2 for i in range (len (y))])
    # implement the algorithm's iteration of IGD. Your result should return the the final xk
    # at the last iteration and also the history of objective function at each xk.
    return z





def IGD_wo_task1(y):
    n = len(y)
    gamma=np.zeros(n)
    x=np.zeros(n)
    z = np.zeros (len (x))
    ordering = np.random.choice(n, n, replace=False)
    # implement the algorithm's iteration of IGD. Your result should return the the final xk
    # at the last iteration and also the history of objective function at each xk.

    for k in range(len(y)):
        for ordering in range(len(y)):
            gamma[k] = 1/(k+1)
            x[k]=x[k]-gamma[k]*(x[k]-y[ordering])
            z[k] = (1 / 2) * sum ([(x[k] - y[i]) ** 2 for i in range (len (y))])
    # implement the algorithm's iteration of IGD. Your result should return the the final xk
    # at the last iteration and also the history of objective function at each xk.
    return z





##########

def IGD_wr_task2(y):
    n = len (y)
    gamma = np.zeros (n)
    beta = np.random.uniform (1, 2, n)
    x = np.zeros (n)
    z = np.zeros (len (x))
    ordering = np.random.choice(n, n, replace=True)
    # implement the algorithm's iteration of IGD. Your result should return the the final xk
    # at the last iteration and also the history of objective function at each xk.

    for k in range(len(y)):
        for ordering in range(len(y)):
            gamma[k] = 0.95*min(1/beta)
            x[k]=x[k]-gamma[k]*(x[k]-y[k])*beta[ordering]
            z[k] = (1 / 2) * sum ([beta[i]*(x[k]-y[k])**2 for i in range (len (beta))])
    # implement the algorithm's iteration of IGD. Your result should return the the final xk
    # at the last iteration and also the history of objective function at each xk.
    return z


# IGD, the ordering is not permitted to have replacement.
#
#
def IGD_wo_task2(y):
    n = len(y)
    gamma=np.zeros(n)
    beta = np.random.uniform(1,2,n)
    x=np.zeros(n)
    z = np.zeros (len (x))
    ordering = np.random.choice(n, n, replace=False)
    # implement the algorithm's iteration of IGD. Your result should return the the final xk
    # at the last iteration and also the history of objective function at each xk.

    for k in range(len(y)):
        for ordering in range(len(y)):
            gamma[k] = 0.95*min(1/beta)
            x[k]=x[k]-gamma[k]*(x[k]-y[k])*beta[ordering]
            z[k] = (1 / 2) * sum ([beta[i]*(x[k]-y[k])**2 for i in range (len (beta))])
    # implement the algorithm's iteration of IGD. Your result should return the the final xk
    # at the last iteration and also the history of objective function at each xk.
    return z
#print(IGD_wo_task2(y1))

#######


def generate_problem_task3(m, n, rho):
    A = np.random.normal(0., 1.0, (m, n))
    x = np.random.random(n) # uniform in (0,1)
    w = np.random.normal(0., rho, m)
    y = A@x + w
    return A, x, y



# We generate the problem with 200x100 matrix. rho as 0.01.
#
A, xstar, y = generate_problem_task3(20, 100, 0.01)

gamma=10e-3
#
# IGD, the ordering is permitted to have replacement.




#
def IGD_wr_task3(y, A):
    m = len (y)
    for col in A.T:
        at=col
    x = np.zeros (A.shape[1])
    z = np.zeros (len (x))
    ordering = np.random.choice (m, m, replace=True)
    # implement the algorithm's iteration of IGD. Your result should return the the final xk
    # at the last iteration and also the history of objective function at each xk.

    for k in range (len (y)):
        for ordering in range (len (y)):
            x[k] = x[k] - gamma* (at.T[ordering]*x[k] - y[ordering]) * at[ordering]
            z[k] =sum ([(at.T[i]*x[k]-y[i])**2 for i in range (m)])
            fel = abs ((x[k] - xstar))
    # implement the algorithm's iteration of IGD. Your result should return the the final xk
    # at the last iteration and also the history of objective function at each xk.
    return x,fel
# IGD, the ordering is not permitted to have replacement.

#
def IGD_wo_task3(y, A):
    m = len (y)
    for col in A.T:
        at=col
    x = np.zeros (A.shape[1])
    z = np.zeros (len (x))

    ordering = np.random.choice (m, m, replace=False)
    # implement the algorithm's iteration of IGD. Your result should return the the final xk
    # at the last iteration and also the history of objective function at each xk.
    for k in range (len (y)):
        for ordering in range (len (y)):
            x[k] = x[k] - gamma* (at.T[ordering]*x[k] - y[ordering]) * at[ordering]
            z[k] =sum ([(at.T[i]*x[k]-y[i])**2 for i in range (m)])
            fel = abs ((x[k] - xstar))
    # implement the algorithm's iteration of IGD. Your result should return the the final xk
    # at the last iteration and also the history of objective function at each xk.
    return x,fel
z1,fel1=IGD_wr_task3(y,A)
z2,fel2=IGD_wo_task3(y,A)


figure=plt.figure()
axes1=figure.add_subplot(2,1,1)
axes2=figure.add_subplot(2,1,2)



axes1.plot(fel1,z1)
axes2.plot(fel2,z2)
plt.show()