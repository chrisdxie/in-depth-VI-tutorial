#__author__ = 'ChrisXie'

""" This file implements variational inference for the d-dimensional Bayesian GMM with known variance.
"""

from __future__ import division
from common_utils import *
import time

def run_variational_inference():

    data = np.loadtxt("data.txt")
    if d == 1: # Hack because np.savetxt() for a 2d array of only one column/row saves it as a 1d array
        data = np.atleast_2d(data).T

    iter_max = 50
    ELBO_times, ELBO_values, var_dists = variational_inference(data, iter_max)

    print "ELBO Values:\n", ELBO_values[0], ELBO_values[-1]

    # Plot ELBO values
    figure(1)
    plot(ELBO_times, ELBO_values)
    xlabel('Time (s)')
    ylabel('ELBO')
    title('ELBO values over time')

    # Plot data
    figure(2)
    alphas = var_dists["pi"][0].alpha
    if d == 1:
        n, bins, patches = hist(data, bins=100, normed=True)
        for k in range(K):
            mu = var_dists["mu"][k].expected_sufficient_stats()
            hist(mu[0], bins=bins, normed=False, weights=[max(n)], color='r', alpha=alphas[k]/sum(alphas) / np.max(alphas/sum(alphas)))
    elif d == 2:
        plot(data[:,0], data[:,1], 'o', mfc="None", mec='g')
        for k in range(K):
            mu = var_dists["mu"][k].expected_sufficient_stats()
            plot(mu[0], mu[1], 'ro', alpha=alphas[k]/sum(alphas) / np.max(alphas/sum(alphas)))

    show()

    return

def variational_inference(data, iter_max):

    # Debugging
    iteration_verbosity = 10

    # Correction term (used for SVI, not traditional VI. It is set to 1 for traditional VI)
    correction = 1

    # Repeat loop
    # iter_max = 50
    num_iters = 0

    # Initialize variational distributions
    var_dists = {}
    var_dists["pi"] = [Dirichlet(alpha_0 * np.ones(K))]
    var_dists["mu"] = []
    var_dists["z"] = []
    for k in range(K):
        mu_init = np.random.multivariate_normal(mu_0, Sigma_tau)  # randomly initialize the mean, keep Sigma_tau
        var_dists["mu"].append(Gaussian(np.concatenate((Sigma_tau_inv.dot(mu_init),
                                                        -1/2 * Sigma_tau_inv.reshape(-1) ))))
    for n in range(N):
        var_dists["z"].append(Categorical(1/K * np.ones(K)))

    # Keep track of ELBO values and times
    ELBO_values = []
    ELBO_times = []
    start_time = time.time()

    while num_iters < iter_max:

        if num_iters % 10 == 0:
            print "Beginning iteration:",  num_iters

        # from IPython import embed; embed()

        # Update all local variational parameters
        gamma_grad = ELBO_gradient_gamma(var_dists, data)
        for n in range(N):
            q_z_n = var_dists["z"][n]
            temp = q_z_n.gamma + gamma_grad[n,:]  # same as using step size of 1
            temp = exp(temp); temp /= np.sum(temp); temp = log(temp)  # normalize
            q_z_n.update_natural_parameters(temp)

        # from IPython import embed; embed()

        # Update all global variational parameters
        nu_grad = ELBO_gradient_nu(var_dists, correction)
        q_pi = var_dists["pi"][0]
        q_pi.update_natural_parameters(q_pi.nu + nu_grad)

        # from IPython import embed; embed()

        lambda_grad = ELBO_gradient_lambda(var_dists, data, correction)
        # from IPython import embed; embed()
        for k in range(K):
            q_mu_k = var_dists["mu"][k]
            q_mu_k.update_natural_parameters(q_mu_k.lambdas + lambda_grad[k,:])

        # Debugging
        if num_iters % iteration_verbosity == 0:
            alphas = var_dists["pi"][0].alpha
            pi_string = "["
            for k in range(K):
                pi_string += str(alphas[k]/sum(alphas)) + ", "
            pi_string = pi_string[:-2] + "]"
            print "E[pi]:", pi_string
            for q_mu_k in var_dists["mu"]:
                print q_mu_k.canonical_parameters()

        # Compute ELBO
        ELBO_values.append(compute_ELBO(var_dists, data))
        ELBO_times.append(time.time() - start_time)

        num_iters += 1

    return ELBO_times, ELBO_values, var_dists


if __name__ == '__main__':
    run_variational_inference()