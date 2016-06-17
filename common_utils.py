""" This file contains the functions that each algorithm will use.

    Recall that this example is based on a Bayesian GMM with known variance.
"""

from __future__ import division
from pylab import *
from scipy.special import psi, gammaln

__author__ = 'ChrisXie'

# Number of data points N
N = 1000

# Dimensionality of data points
d = 1

# Number of clusters k
K = 3

# covariance matrix of data
Sigma = 1 * np.eye(d)
Sigma_inv = np.linalg.inv(Sigma)  # Precomputation

# Hyperparameters for pi
alpha_0 = 1

# Hyperparameters for mu_k
if d == 1:
    mu_0 = np.array([0])
elif d == 2:
    mu_0 = np.array([0,0])
Sigma_0 = 3 * np.eye(d) # covariance matrix of prior on mu_k
Sigma_0_inv = np.linalg.inv(Sigma_0)  # Precomputation


##########  Variational Distributions ##########

class Categorical(object):

    def __init__(self, gamma):
        if gamma.shape[0] != K:
            raise Exception("Gamma vector must be K-dimensional!")
        self.gamma = gamma # gamma should be a K-dimensional np.array

    def update_natural_parameters(self, gamma):
        self.gamma = gamma

    # Return Prob(X_k = 1) for all k
    def expected_sufficient_stats(self):
        return np.exp(self.gamma)

    def entropy(self):
        return -1 * self.gamma.dot(exp(self.gamma))


class Gaussian(object):

    def __init__(self, lambdas):
        # Natural parameters to a d dimensional Gaussian distribution
        if lambdas.shape[0] != d+pow(d,2):
            raise Exception("Lambda vector must be (d + d^2)-dimensional!")
        self.lambdas = lambdas

    def update_natural_parameters(self, lambdas):
        self.lambdas = lambdas

    def canonical_parameters(self):
        lambda_1 = self.lambdas[:d]
        lambda_2 = self.lambdas[d:].reshape([d,d])
        Sigma = -0.5 * np.linalg.inv(lambda_2)
        mu = Sigma.dot( lambda_1 )
        return mu, Sigma

    # Return E[X], E[XX^T] (in stacked vector form)
    def expected_sufficient_stats(self):
        mu, Sigma = self.canonical_parameters()
        temp_0 = mu
        temp_1 = (Sigma + np.outer(mu, mu)).reshape(-1)
        return np.concatenate((temp_0, temp_1))

    def entropy(self):
        mu, Sigma = self.canonical_parameters()
        H = d/2 * np.log(2 * np.pi * np.exp(1)) + 1/2 * np.log(np.linalg.det(Sigma))
        return H


class Dirichlet(object):

    def __init__(self, alpha):
        if alpha.shape[0] != K:
            raise Exception("Alpha vector must be K-dimensional!")
        self.alpha = alpha # alpha should be a K-dimensional np.array
        self.nu = alpha - 1

    def update_natural_parameters(self, nu):
        self.nu = nu
        self.alpha = nu + 1

    # Return E[ln X_k] for all k
    def expected_sufficient_stats(self):
        suff_stats = psi(self.alpha) - psi(np.sum(self.alpha))
        return np.array(suff_stats)

    def log_normalizer(self):
        A = np.sum(gammaln(self.alpha))
        A -= gammaln(np.sum(self.alpha))
        return A

    def entropy(self):
        exp_suff_stats = self.expected_sufficient_stats()
        return -1 * self.nu.dot(exp_suff_stats) + self.log_normalizer()


########## Computing the ELBO ##########

def compute_ELBO(var_dists, data):
    """
    :param var_dists: the variational distributions for all of the hidden variables
                                      should be a dictionary
    :return: the value of the ELBO
             should be a scalar
    """

    # Precompute some terms here
    pi_exp_suff_stats = var_dists["pi"][0].expected_sufficient_stats()
    mu_exp_suff_stats = []
    for k in range(K):
        mu_exp_suff_stats.append(var_dists["mu"][k].expected_sufficient_stats())
    mu_exp_suff_stats = np.array(mu_exp_suff_stats)

    ELBO = 0


    ### Cross entropy terms ###

    # E_q[ln P(\pi)]
    C_alpha_0 = np.sum(gammaln(alpha_0 * np.ones(K)))
    C_alpha_0 -= gammaln(np.sum(alpha_0 * np.ones(K)))
    C_alpha_0 *= -1
    ELBO += C_alpha_0 + (alpha_0 - 1) * np.sum(pi_exp_suff_stats)

    # E_q[ln P(\mu_k)]
    ELBO += - K * d/2 * log(2 * np.pi) - K * 1/2 * log(np.linalg.det(Sigma_0))
    for k in range(K):
        ELBO += -1/2 * (Sigma_0_inv.reshape(-1).dot(mu_exp_suff_stats[k,d:])
                        - 2 * mu_0.dot( Sigma_0_inv ).dot( mu_exp_suff_stats[k,:d] )
                        + mu_0.dot( Sigma_0_inv ).dot( mu_0 ))

    # E_q[ln P(z | \pi)]
    for n in range(N):
        z_n_exp_suff_stats = var_dists["z"][n].expected_sufficient_stats()
        ELBO += z_n_exp_suff_stats.dot(pi_exp_suff_stats)

    # E_q[ln P(x | z, mu)]
    ELBO += -K * d/2 * log(2 * np.pi) - K * 1/2 * log(np.linalg.det(Sigma))
    for n in range(N):
        z_n_exp_suff_stats = var_dists["z"][n].expected_sufficient_stats()
        for k in range(K):
            x_n = data[n,:]
            temp = z_n_exp_suff_stats[k] * (x_n.dot(Sigma_inv).dot(x_n)
                                            - 2 * x_n.dot(Sigma_inv).dot(mu_exp_suff_stats[k, :d])
                                            + Sigma_inv.reshape(-1).dot(mu_exp_suff_stats[k, d:]))
            ELBO += -1/2 * temp


    ### Entropy Terms ###

    ELBO += var_dists["pi"][0].entropy()
    for k in range(K):
        ELBO += var_dists["mu"][k].entropy()
    for n in range(N):
        ELBO += var_dists["z"][n].entropy()

    return ELBO



########## ELBO Gradients ##########

# Global updates

def ELBO_gradient_nu(var_dists, correction):
    """
    :param var_dists: the variational distributions for the hidden variables
                                      should be a dictionary
    :param correction: scalar correction term (used for SVI. for traditional VI, it is equal to 1)
    :return: the gradient of the ELBO w.r.t. nu
             returns an np.array
    """

    nu = var_dists["pi"][0].nu

    # Compute \sum_n e^{\gamma_{nk}}
    q_z = var_dists["z"]
    matrix_of_exp_suff_stats = []
    for q_z_n in q_z:
        matrix_of_exp_suff_stats.append(q_z_n.expected_sufficient_stats())
    matrix_of_exp_suff_stats = np.array(matrix_of_exp_suff_stats)

    return alpha_0 + correction * matrix_of_exp_suff_stats.sum(axis=0) - 1 - nu

def ELBO_gradient_lambda(var_dists, data, correction):
    """
    :param var_dists: the variational distributions for the hidden variables
                                      should be a dictionary
    :param data: the data for the problem
    :param correction: scalar correction term (used for SVI. for traditional VI, it is equal to 1)
    :return: the gradient of the ELBO w.r.t. lambda_k for all k
             returns an np.array of dimension K x 2
    """
    q_z = var_dists["z"]
    matrix_of_exp_suff_stats = []
    for q_z_n in q_z:
        matrix_of_exp_suff_stats.append(q_z_n.expected_sufficient_stats())
    matrix_of_exp_suff_stats = np.array(matrix_of_exp_suff_stats)

    # Compute \sum_n e^{\gamma_{nk}} x_n, \sum_n e^{\gamma_{nk}}
    sum_gamma_x = correction * matrix_of_exp_suff_stats.T.dot(data)
    sum_gamma = correction * matrix_of_exp_suff_stats.sum(axis=0)

    gradients = []
    for k in range(K):
        lambda_k = var_dists["mu"][k].lambdas
        temp_0 = (Sigma_0_inv.dot(mu_0) + Sigma_inv.dot(sum_gamma_x[k])) - lambda_k[:d]
        temp_1 = -1/2 * (Sigma_0_inv + Sigma_inv * sum_gamma[k]).reshape(-1) - lambda_k[d:]
        # from IPython import embed; embed()
        gradients.append(np.concatenate((temp_0, temp_1)))
    gradients = np.array(gradients)
    return gradients


# Local updates

def ELBO_gradient_gamma(var_dists, data):

    """
    :param var_dists: the variational distributions for the hidden variables
                                      should be a dictionary
    :param data: the data for the problem
    :return: the gradient of the ELBO w.r.t. gamma_n for all n
             returns an np.array of dimension N x K
    """

    q_pi = var_dists["pi"][0]
    q_mu = var_dists["mu"]

    first_term = q_pi.expected_sufficient_stats()

    gradients = []
    for n, q_z_n in enumerate(var_dists["z"]):
        gamma_n = q_z_n.gamma
        x_n = data[n,:]

        second_term = []
        for k in range(K):
            mu_k_exp_suff_stats = q_mu[k].expected_sufficient_stats()
            temp = x_n.dot(Sigma_inv).dot(x_n)\
                   - 2 * x_n.dot(Sigma_inv).dot(mu_k_exp_suff_stats[:d])\
                   +  Sigma_inv.reshape(-1).dot(mu_k_exp_suff_stats[d:])
            temp *= -1/2
            second_term.append(temp)
        second_term = np.array(second_term)

        gradients.append(first_term + second_term - gamma_n)

    return np.array(gradients)


