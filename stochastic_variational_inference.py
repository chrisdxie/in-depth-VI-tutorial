#__author__ = 'ChrisXie'

""" This file implements stochastic variational inference for the d-dimensional Bayesian GMM with known variance.
"""

from __future__ import division
from common_utils import *
import time

def run_stochastic_variational_inference():

    data = np.loadtxt("data.txt")
    if d == 1: # Hack because np.savetxt() for a 2d array of only one column/row saves it as a 1d array
        data = np.atleast_2d(data).T

    iter_max = 500
    minibatch_size = 20
    ELBO_times, ELBO_values, var_dists = stochastic_variational_inference(data, iter_max, minibatch_size)

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

# Step size
def rho(t):
    delay = 100
    forget_rate = 1
    return pow(1/(t + delay), forget_rate)

def stochastic_variational_inference(data, iter_max, minibatch_size):

    # Debugging
    iteration_verbosity = 50

    # Correction term for SVI
    correction = N

    # Repeat loop
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

    # Use this for SVI gradients
    incomplete_var_dists = {}
    incomplete_var_dists["pi"] = var_dists["pi"]
    incomplete_var_dists["mu"] = var_dists["mu"]

    # Keep track of ELBO values
    ELBO_values = []
    ELBO_times = []
    start_time = time.time()

    while num_iters < iter_max:

        if num_iters % iteration_verbosity == 0:
            print "Beginning iteration:",  num_iters

        # Select minibatch (and their indices)
        minibatch_indices = []
        for i in range(minibatch_size):
            minibatch_indices.append(np.random.randint(N))

        # from IPython import embed; embed()

        # Update randomly sampled subset of local variational parameters
        for i in minibatch_indices:
            incomplete_var_dists["z"] = [var_dists["z"][i]]
            gamma_grad = ELBO_gradient_gamma(incomplete_var_dists, np.array([data[i,:]]))

            q_z_i = var_dists["z"][i]
            temp = q_z_i.gamma + gamma_grad[0,:]  # same as using step size of 1
            temp = exp(temp); temp /= np.sum(temp); temp = log(temp)  # normalize
            q_z_i.update_natural_parameters(temp)

        # from IPython import embed; embed()

        # Update all global variational parameters
        nu_grads = []
        for i in minibatch_indices:
            incomplete_var_dists["z"] = [var_dists["z"][i]]
            grad = ELBO_gradient_nu(incomplete_var_dists, correction)
            nu_grads.append(grad)
        # Average the intermediate global parameters for pi
        nu_grads = np.array(nu_grads)
        nu_grad = 1/minibatch_size * np.sum(nu_grads, axis=0)
        q_pi = var_dists["pi"][0]
        q_pi.update_natural_parameters((1 - rho(num_iters)) * q_pi.nu + rho(num_iters) * nu_grad)


        lambda_grads = []
        for i in minibatch_indices:
            incomplete_var_dists["z"] = [var_dists["z"][i]]
            grad = ELBO_gradient_lambda(incomplete_var_dists, np.array([data[i,:]]), correction)
            lambda_grads.append(grad)
        # Average the intermediate global parameters for lambda
        lambda_grads = np.array(lambda_grads)
        lambda_grad = 1/minibatch_size * np.sum(lambda_grads, axis=0)
        for k in range(K):
            q_mu_k = var_dists["mu"][k]
            q_mu_k.update_natural_parameters((1 - rho(num_iters)) * q_mu_k.lambdas + rho(num_iters) * lambda_grad[k,:])

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
    run_stochastic_variational_inference()