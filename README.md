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
#     σ be the standard deviation of the measurement noise

# prepare experimental data
θ_g = rand(d)                    # underlying parameter
X   = rand(d, n)                 # features
y   = vec(θ_g'X) + σ * randn(n)  # responses

# initialize solution
θ_0 = zeros(d)

# optimize
θ = sgd(sqrloss!, θ_0,
    minibatch_seq(X, y, 10),       # a stream of mini-batches of size 10
    lrate=t->1.0 / (100.0 + t),  # configure the policy to compute learning rate
    cbctrl=ByInterval(5),     # invoke the callback every 5 iteration
    callback=simple_trace     # callback: print the optimization trace when invoked
)

```

From this example, we can see that an SGD optimization procedure involves multiple aspects:

- The optimization algorithm: ``sgd`` indicates the use of the standard SGD. The package also provides other algorithms.

- The loss function: ``sqrloss!`` is a functor, which indicates that we use *squared loss*.

- The data stream.

  **Note:** This package provides various ways to configure the data stream. For example, data can be supplied on a per-sample basis in any given order, or via mini-batches.

- The callback mechanism that enables the interoperability with the world. Particularly, we use an ``cbctrl`` option to control how frequently the callback is invoked, and the ``callback`` option to actually supply the callback.

---

## Algorithms

This package provides the following algorithms.

#### Stochastic Methods (per-sample or per-mini-batch updates)

- [x] Stochastic Gradient Descent
- [ ] Accelerated Stochastic Gradient Descent
- [ ] Stochastic Proximal Gradient Descent

#### Parallel/Distributed Methods

- [ ] Hogwild!
- [ ] Parallel Alternate Direction Methods of Multipliers (ADMM)
- [ ] ADMM with Variable Splitting
