# In-Depth Variational Inference Tutorial

## Common Utils

In `common_utils.py`, there are some parameters you may want to set. It contains `N, d, K` (number of data points, dimension, number of clusters) and some hyperparameters. The code should be able to handle any value of `d`, but right now we are plotting stuff, and the code can only plot up to 2 dimensions. So if `d > 2`, you should turn off the plotting.

## Data generation

After having set your parameters, you can generate data using `python data_generator.py`. You will get print outs of parameter values and plots of the data you just generated. It saves the data to a file called `data.txt`. If you look inside `data_generator.py`, you can always figure out how to plot data from `data.txt` if you generated it a while back and forgot what it looks like.

## Running the algorithms

To run the VI/SVI algorithms, simply run:

* `$:~ python variational_inference.py`
* `$:~ python stochastic_variational_inference.py`.

You must make sure that `data.txt` is in the current directory. Look through the code for more details. We also include a comparison script to compare batch VI vs. minibatch SVI. You can run this by running:

* `$:~ python compare_VI_SVI.py`

These files will give plots that were shown in the tutorial.
