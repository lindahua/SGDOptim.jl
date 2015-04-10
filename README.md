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

# prepare experimental data
d = length(theta_g)
X = randn(d, n)
y = vec(theta_g'X) + σ * randn(n)

# initialize solution
theta_0 = zeros(d)

# optimize
sol = sgd(sqrloss!, theta_0,
    SampleSeq(X, y, randperm(n)),  # supply a stream of samples, using random permuted order
    cbctrl=ByInterval(100),        # invoke the callback every 100 iteration
    callback=gtcompare_trace(theta_g)  # callback: print the optimization trace when invoked
)

```

From this example, we can see that an SGD optimization procedure involves multiple aspects:

- The optimization algorithm: ``sgd`` indicates the use of the standard SGD. The package also provides other algorithms.

- The loss function: ``sqrloss!`` is a functor, which indicates that we use *squared loss*.

- The data stream, which is given by both the sample set ``X`` and ``y``, as well as the order of supplying the samples.

  **Note:** with the streaming facilities provided by the package, one can supply the data as mini-batches instead of on an per-sample basis.

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
