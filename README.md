# SGDOptim

A Julia package for Stochastic Gradient Descent (SGD) and its variants.

[![Build Status](https://travis-ci.org/lindahua/SGDOptim.jl.svg?branch=master)](https://travis-ci.org/lindahua/SGDOptim.jl)

---

With the advent of *Big Data*, *Stochastic Gradient Descent (SGD)* has become increasingly popular in recent years, especially in machine learning and related areas. This package implements the SGD algorithm and its variants under a generic setting to facilitate the use of SGD in practice.

Here is an [example](http://nbviewer.ipython.org/github/lindahua/SGDOptim.jl/blob/master/example.ipynb) that demonstrates the use of this package in solving a ridge regression problem.


### Optimization Algorithms

This package depends on [EmpiricalRisks.jl](https://github.com/lindahua/EmpiricalRisks.jl), which provides the basic components, including *predictors*, *loss functions*, and *regularizers*.

On top of that, we provide a variety of algorithms, including SGD and its variants, and you may choose one that is suitable for your need:

**For streaming settings:**

- [x] Stochastic Gradient Descent
- [ ] Accelerated Stochastic Gradient Descent
- [ ] Stochastic Proximal Gradient Descent

**For distributed settings:**

- [ ] Parallel Alternate Direction Methods of Multipliers (ADMM)
- [ ] ADMM with Variable Splitting

**Learning rate:**

The setting of the *learning rate* has significant impact on the algorithm's behavior. This package allows the learning rate setting to be provided as a function on ``t`` as a keyword argument.

The default setting is ``t -> 1.0 / (1.0 + t)``.


### Key Functions

- **sgd**(rmodel, theta, stream; ...)

  Performs stochastic gradient descent to solve a (regularized) risk minimization problem.

  |  params   |  descriptions |
  | --------- | ------------- |
  | `rmodel`  | the risk model, which can be constructed using `riskmodel` method.  |
  | `theta`   | The initial guess of the model parameter. |
  | `stream`  | The input data stream. |

  This function also accepts keyword arguments:

  | params       | descriptions |
  | ------------ | ------------ |
  | `reg`        | the regularizer (default = `ZeroReg()`, means no regularization). |
  | `lrate`      | the learning rate rule, which should be a function of `t` (default as mentioned above). |
  | `cbinterval` | the interval of invoking the callback, *i.e.* the function invokes the callback every `cbinterval` iterations. (default is `0`, meaning that it never invokes the callback). |



### Interoperability

We allow the optimization procedure to interoperate with the rest of the world, through the callback mechanism.

The user can supply a callback function via the ``callback`` keyword argument, which will be invoked as the optimization proceeds. We understand that invoking the callback too frequently may incur considerable overhead in certain situations, and hence provide a ``cbinterval`` option, which allows the user to specify how frequently the callback should be invoked.
