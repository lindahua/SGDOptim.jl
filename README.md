# SGDOptim

A Julia package for Stochastic Gradient Descent (SGD) and its variants.

[![Build Status](https://travis-ci.org/lindahua/SGDOptim.jl.svg?branch=master)](https://travis-ci.org/lindahua/SGDOptim.jl)

---

With the advent of *Big Data*, *Stochastic Gradient Descent (SGD)* has become increasingly popular in recent years, especially in machine learning and related areas. This package implements the SGD algorithm and its variants under a generic setting to facilitate the use of SGD in practice.

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
θ = sgd(sqrloss, θ_0,
    minibatch_seq(X, y, 10),     # a stream of mini-batches of size 10
    lrate=t->1.0 / (100.0 + t),  # configure the policy to compute learning rate
    cbinterval=5,             # invoke the callback every 5 iterations
    callback=simple_trace     # callback: print the optimization trace when invoked
)

```

This example shows several aspects involved in an SGD optimization procedure:


### Optimization Algorithms

Here, we call the ``sgd`` function, which implements the standard SGD algorithm. This package provides a variety of algorithms:

**For streaming settings:**

- [x] Stochastic Gradient Descent
- [ ] Accelerated Stochastic Gradient Descent
- [ ] Stochastic Proximal Gradient Descent

**For distributed settings:**

- [ ] Hogwild!
- [ ] Parallel Alternate Direction Methods of Multipliers (ADMM)
- [ ] ADMM with Variable Splitting


### Loss Functions

Here, we use ``sqrloss`` to indicate the use of *Squared loss*, which is a popular choice for linear regression. This package provides a collection of loss functions:

- [x] Squared loss
- [ ] Hinge loss
- [ ] Logistic loss
- [ ] Multinomial logistic loss
- [ ] L1-norm quantile loss

In addition, the package specifies a uniform interface for users to implement and use their customized loss functions.

### Learning Rate

The setting of the *learning rate* has significant impact on the algorithm's behavior. This package allows the learning rate setting to be provided as a function on ``t`` as a keyword argument.

The default setting is ``t -> 1.0 / (1.0 + t)``.

### Interoperability

We allow the optimization procedure to interoperate with the rest of the world, through the callback mechanism.

The user can supply a callback function via the ``callback`` keyword argument, which will be invoked as the optimization proceeds. We understand that invoking the callback too frequently may incur considerable overhead in certain situations, and hence provide a ``cbinterval`` option, which allows the user to specify how frequently the callback should be invoked.
