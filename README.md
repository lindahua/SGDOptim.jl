# SGDOptim

A Julia package for Stochastic Gradient Descent (SGD) and its variants.

[![Build Status](https://travis-ci.org/lindahua/SGDOptim.jl.svg?branch=master)](https://travis-ci.org/lindahua/SGDOptim.jl)

---

With the advent of *Big Data*, *Stochastic Gradient Descent (SGD)* has become increasingly popular in recent years, especially in machine learning and related areas. This package implements the SGD algorithm and its variants under a generic setting to facilitate the use of SGD in practice.

Here is an [example](http://nbviewer.ipython.org/github/lindahua/SGDOptim.jl/blob/master/example.ipynb) that demonstrates the use of this package in solving a ridge regression problem.


## Optimization Algorithms

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


#### Key Functions


- **sgd**(rmodel, theta, stream; ...)

  Performs stochastic gradient descent to solve a (regularized) risk minimization problem.

  |  params   |  descriptions |
  | --------- | ------------- |
  | `rmodel`  | the risk model, which can be constructed using [riskmodel](http://empiricalrisksjl.readthedocs.org/en/latest/riskmodels.html#risk-models) method.  |
  | `theta`   | The initial guess of the model parameter. |
  | `stream`  | The input data stream. (See the *Streams* section below for details) |

  This function also accepts keyword arguments:

  | params       | descriptions |
  | ------------ | ------------ |
  | `reg`        | the regularizer (default = `ZeroReg()`, means no regularization). See the [documentation on regularizers](http://empiricalrisksjl.readthedocs.org/en/latest/regularizers.html) for details. |
  | `lrate`      | the learning rate rule, which should be a function of `t` (default as mentioned above). |
  | `callback`   | the callback function, which will be invoked during iterations. default is ``simple_trace``. See the *Callbacks* section below for detail. |
  | `cbinterval` | the interval of invoking the callback, *i.e.* the function invokes the callback every `cbinterval` iterations. (default is `0`, meaning that it never invokes the callback). |



## Streams

Unlike conventional methods, SGD and its variants look at a single sample or a small batch of samples at each iteration. In other words, data are viewed as a stream of samples or minibatches.

This package provides a variety of ways to construct data streams. Each data stream is essentially an iterator that implements the ``start``, ``done``, and ``next`` methods (see [here]( <http://julia.readthedocs.org/en/latest/stdlib/collections/#iteration) for details of Julia's iteration patterns). Each item from a data stream can be either a sample (as a pair of input and output) or a mini-batch (as a pair of multi-input array and multi-output array).

**Note:** All SGD algorithms in this package support both sample streams and mini-batch streams. At each iteration, the algorithm works on a single item from the stream, which can be either a sample or a mini-batch.


The package provides several methods to construct streams of samples or minibatches.

- **sample_seq**(X, Y[, ord])

    Wrap an input array ``X`` and an output array ``Y`` into a stream of individual samples.

    Each item of the stream is a pair, comprised of an item from ``X`` and a corresponding item from ``Y``. If ``X`` is a vector, then each item of ``X`` is a scalar, if ``X`` is a matrix, then each item of ``X`` is a column vector. The same applies to ``Y``.

    The ``ord`` argument is an instance of ``AbstractVector`` that specifies the order in which the samples are scanned. If ``ord`` is omitted, it is, by default, set to the natural order, namely,
    ``1:n``, where ``n`` is the number of samples in the data set.

- **minibatch_seq**(X, Y, bsize[, ord])

    Wrap an input array ``X`` and an output array ``Y`` into a stream of mini-batches of size ``bsize`` or smaller.

    For example, if ``X`` and ``Y`` have ``28`` samples, by setting ``bsize`` to ``10``, we partition the data set into three minibatches, respectively corresponding to the indices ``1:10``, ``11:20``, and ``21:28``.

    The ``ord`` argument specifies the order in which the mini-batches are used. For example, if ``ord`` is set to ``[3, 2, 1]``, it first takes the 3rd batch, then 2nd, and finally 1st. If ``ord`` is omitted, it is, by default, set to the natural order, namely, ``1:m``, where ``m`` is the number of mini-batches.


## Callbacks

The algorithms provided in this package interoperate with the rest of the world through *callbacks*. In particular, it allows a third party (*e.g.* a higher-level script, a user, a GUI, etc) to monitor the progress of the optimization and take proper actions.

Generally, a *callback* is an arbitrary function (or closure) that can be called in the following way:

```
callback(theta, t, n, v)
```

| params  | descriptions |
| ------- | ------------ |
| `theta` | The current solution. |
| `t`     | The number of elapsed iterations. |
| `n`     | The number of samples that have been used. |
| `v`     | The objective value of the last item, which can be an objective evaluated on a single sample or the total objective value evaluated on the last batch of samples. |

The package already provides some callbacks for simple use:

- `simple_trace`

    Simply print the optimization trace, including the number of iterations, and the average loss of the last iteration.

    This is the default choice for most algorithms.


- `gtcompare_trace(theta_g)`

    In addition to printing the optimization trace, it also computes and shows the deviation from a given oracle ``theta_g``.

    **Note:** ``gtcompare_trace`` is a high-level function, and ``gtcompare_trace(theta_g)`` produces a callback function.
