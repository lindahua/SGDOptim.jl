# SGDOptim

A Julia package for Stochastic Gradient Descent (SGD) and its variants.

[![Build Status](https://travis-ci.org/lindahua/SGDOptim.jl.svg?branch=master)](https://travis-ci.org/lindahua/SGDOptim.jl)

---

With the advent of *Big Data*, *Stochastic Gradient Descent (SGD)* has become increasingly popular in recent years, especially in machine learning and related areas. This package implements the SGD algorithm and its variants under a generic setting to facilitate the use of SGD in practice.

Here is an [example](http://nbviewer.ipython.org/github/lindahua/SGDOptim.jl/blob/master/example.ipynb) that demonstrates the use of this package in solving a ridge regression problem.


## Overview

Generally, SGD optimization is a problem that involves multiple aspects:

- Optimization problem: predictor, loss function, and regularizer
- Optimization algorithm
- Data: sample sequence or mini-batch?
- Interoperability with the world (*e.g.* monitor the progress)

Below is an overview of what we provide in this package:


### Optimization Problem

A regularized risk minimization problem is generally comprised of three parts:

**Predictor:**

- [x] Linear predictor
- [x] Affine predictor, *i.e.* linear predictor with a bias term
- [x] Multivariate linear predictor
- [x] Multivariate affine predictor

**Loss function:**

- [x] Squared loss
- [x] Hinge loss
- [x] Logistic loss
- [ ] Multinomial logistic loss
- [ ] L1-norm quantile loss

**Regularizer:**

- [x] No regularization
- [x] Squared L2-norm
- [x] L1-norm (*e.g.* LASSO)
- [x] Elastic Net
- [ ] Grouped LASSO
- [ ] Fused LASSO


### Optimization Algorithms

We provide a variety of algorithms, including SGD and its variants, and you may choose one that is suitable for your need:

**For streaming settings:**

- [x] Stochastic Gradient Descent
- [ ] Accelerated Stochastic Gradient Descent
- [ ] Stochastic Proximal Gradient Descent

**For distributed settings:**

- [ ] Hogwild!
- [ ] Parallel Alternate Direction Methods of Multipliers (ADMM)
- [ ] ADMM with Variable Splitting

**Learning rate:**

The setting of the *learning rate* has significant impact on the algorithm's behavior. This package allows the learning rate setting to be provided as a function on ``t`` as a keyword argument.

The default setting is ``t -> 1.0 / (1.0 + t)``.


### Interoperability

We allow the optimization procedure to interoperate with the rest of the world, through the callback mechanism.

The user can supply a callback function via the ``callback`` keyword argument, which will be invoked as the optimization proceeds. We understand that invoking the callback too frequently may incur considerable overhead in certain situations, and hence provide a ``cbinterval`` option, which allows the user to specify how frequently the callback should be invoked.
