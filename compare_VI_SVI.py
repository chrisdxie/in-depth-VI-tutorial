# __author__ = 'ChrisXie'

from __future__ import division
from common_utils import *
from pylab import *
from IPython import embed
from variational_inference import variational_inference
from stochastic_variational_inference import stochastic_variational_inference


def run_comparison():
    data = np.loadtxt("data.txt")
    if d == 1:  # Hack because np.savetxt() for a 2d array of only one column/row saves it as a 1d array
        data = np.atleast_2d(data).T

    # Error bars
    num_trials_per_minibatch = 5

    # Plotting stuff
    figure(1)

    # Run variational inference and plot
    num_iters_for_VI = 100
    ELBO_times, ELBO_values = [], []
    for j in range(num_trials_per_minibatch):
        print "VI: Trial number {0}".format(j+1)

        _ELBO_times, _ELBO_values, var_dists = variational_inference(data, num_iters_for_VI)
        ELBO_times.append(_ELBO_times)
        ELBO_values.append(_ELBO_values)

    mean_ELBO_times = mean(ELBO_times, axis=0)
    mean_ELBO_values = mean(ELBO_values, axis=0)
    stddev = std(ELBO_values, axis=0)
    errorbar(mean_ELBO_times, mean_ELBO_values, ecolor='y',
             yerr=stddev, fmt='none', errorevery=int(num_iters_for_VI/5))
    plot(mean_ELBO_times, mean_ELBO_values, 'y-', label='batch')


    # Run SVI with different minibatches
    num_iters_for_SVI = 500
    minibatch_sizes = [1, 20, 50]
    line_types_and_colors = ['b-.', 'g--', 'r:']
    for i, size in enumerate(minibatch_sizes):
        ELBO_times, ELBO_values = [], []
        for j in range(num_trials_per_minibatch):

            print "SVI: Batch size {0}. Trial number {1}".format(size, j+1)

            _ELBO_times, _ELBO_values, var_dists = stochastic_variational_inference(data, num_iters_for_SVI, size)
            ELBO_times.append(_ELBO_times)
            ELBO_values.append(_ELBO_values)

        mean_ELBO_times = mean(ELBO_times, axis=0)
        mean_ELBO_values = mean(ELBO_values, axis=0)
        stddev = std(ELBO_values, axis=0)
        errorbar(mean_ELBO_times, mean_ELBO_values, ecolor=line_types_and_colors[i][0],
                 yerr=stddev, fmt='none', errorevery=int(num_iters_for_SVI/5))
        plot(mean_ELBO_times, mean_ELBO_values, line_types_and_colors[i], label='minibatch: ' + str(size))


    legend(loc='lower right')
    xlabel('Time (s)')
    ylabel('ELBO')
    title('ELBO values over time')

    show()


if __name__ == '__main__':
    run_comparison()
