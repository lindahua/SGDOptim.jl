# SGD

A Julia package for Gradient Descent and Stochastic Gradient Descent (SGD)

[![Build Status](https://travis-ci.org/lindahua/SGD.jl.svg?branch=master)](https://travis-ci.org/lindahua/SGD.jl)

---

## Overview

This package aims to provide a rich API as the basis for implementing and testing stochastic gradient descent and its variants, which are among the most widely used techniques for machine learning on large-scale datasets.

Here is an example that illustrates how this package can be used to solve a regression problem:

```julia

# prepare experimental data

# let d be the sample dimension
#     n be the number of samples

theta_g = randn(d)  # ground-truth
X = randn(d, n)
y = vec(theta_g'X) + 0.1 * randn(n)

# initialize solution
theta0 = zeros(d)

# optimize
theta = sgd(
    sqrloss!,    # indicate to use the squared loss
    theta0,      # supply the initial guess
    X, y;        # supply the data set
    cbctrl=ByInterval(100),  # invoke callback every 100 iterations
    callback=simple_trace    # the callback simply prints the optimization trace
)

```

From this example, we can see that an SGD optimization procedure involves multiple aspects:

- The optimization algorithm: ``sgd`` indicates the use of the standard SGD. The package also provides other algorithms.

- The loss function: ``sqrloss!`` is a functor, which indicates that we use *squared loss*.

- The data set, which is given by ``X`` and ``y`` here.

- The callback mechanism that enables the interoperability with the world. Particularly, we use an ``cbctrl`` option to control how frequently the callback is invoked, and the ``callback`` option to actually supply the callback.

---

## Algorithms

This package provides the following algorithms.

#### Conventional Methods

- [ ] Steepest Gradient Descent
- [ ] Nesterov's Accelerated Gradient Descent
- [ ] Proximal Gradient Descent

#### Stochastic Methods (per-sample or per-mini-batch updates)

- [x] Stochastic Gradient Descent
- [ ] Accelerated Stochastic Gradient Descent
- [ ] Stochastic Proximal Gradient Descent

#### Parallel/Distributed Methods

- [ ] Hogwild!
- [ ] Parallel Alternate Direction Methods of Multipliers (ADMM)
- [ ] ADMM with Variable Splitting
