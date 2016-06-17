__author__ = 'ChrisXie'

""" This file generates synthetic 1-dimensional data and stores it in a txt file called:
    data.txt. So original...
"""

#import sys
#import numpy as np
from common_utils import *
from matplotlib import pyplot

def generate_data():

    num_samples = N
    samples = []

    # Sample pi
    pi = np.random.dirichlet(alpha_0 * np.ones(K))

    # Sample mu_k for all k
    mu = []
    for k in range(K):
        candidate = np.random.multivariate_normal(mu_0, Sigma_0)
        mu.append(candidate)
    mu = np.array(mu,dtype = np.float)

    for n in range(num_samples):

        # Sample z_n | pi
        z_n = np.random.multinomial(1,pi) # z_k looks like [0, ..., 0, 1, 0, ..., 0]
        z_n = np.where(z_n == 1) [0][0] # Hack to find index of 1 in z_k

        # Sample x_n | z_n, mu
        x_n = np.random.multivariate_normal(mu[z_n], Sigma)

        samples.append(x_n)

    samples = np.array(samples)

    # Print mu, pi
    print "mu:", mu.T
    print "pi:", pi

    # from IPython import embed; embed()

    plot_data(samples)

def plot_data(samples):

    # Plot data
    if d == 1:
        pyplot.hist(samples, bins=100)
    elif d == 2:
        pyplot.plot(samples[:,0], samples[:,1], 'o', mfc="None", mec='g')
    pyplot.show()

    # Save data
    np.savetxt('data.txt', samples)

def plot_data_from_txt():

    data = np.loadtxt("data.txt")
    if d == 1:  # Hack because np.savetxt() for a 2d array of only one column/row saves it as a 1d array
        data = np.atleast_2d(data).T

    plot_data(data)

if __name__ == '__main__':
    # generate_data()
    plot_data_from_txt()



    