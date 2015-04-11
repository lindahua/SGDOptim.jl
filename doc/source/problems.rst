Construction of Optimization Problems
======================================

This package targets an important family of problems in machine learning and data analytics, namely, *(regularized) empirical risk minimization*. Such problems can generally be expressed as:

.. math::

  \text{minimize } \ \frac{1}{n} \sum_{i=1}^n \mathrm{loss}(f(\theta, x_i), y_i) + \mathrm{reg}(\theta)


This objective is comprised of two parts: the *data terms*, which is the average loss evaluated at all samples, and a *regularization term* that encourages lower model complexity. In particular, each *loss term* compares the output of the predictor ``f`` and a desired output. To sum up, three components are involved: *predictor*, *loss function*, and *regularizer*.

This package provides facilities to construct these components respectively, and the SGD solver will take these components as inputs.

Predictors
-----------

All predictors in this package are organized with the following type hierarchy:

.. code:: julia

  abstract Predictor

  abstract UnivariatePredictor <: Predictor      # produces scalar output
  abstract MultivariatePredictor <: Predictor    # produces vector output


Methods
~~~~~~~~

The following methods are provided for each predictor type. Let ``pred`` be a predictor.

.. function:: predict(pred, theta, x)

  Evaluate and return the predicted output, given the parameter ``theta`` and the input ``x``.

  The form of the output depends on both the predictor type and the input.

  ========================== ============================== ================================
  predictor type               input ``x``                    output
  ========================== ============================== ================================
  ``UnivariatePredictor``      a vector of length ``d``       a scalar
  ``UnivariatePredictor``      a matrix of size ``(d, n)``    a vector of length ``n``
  ``MultivariatePredictor``    a vector of length ``d``       a vector of length ``q``
  ``MultivariatePredictor``    a matrix of size ``(d, n)``    a matrix of size ``(q, n)``
  ========================== ============================== ================================

  **Note:** for multivariate predictors, the output dimension ``q`` need not be equal to
  the input dimension ``d``.


.. function:: scaled_grad!(pred, g, c, theta, x)

  Evaluate scaled gradient(s) at given sample(s), writing the resultant gradient(s) to ``g``.

  When ``x`` is a vector that represents a single sample, it computes ``c`` times the gradient and writes the
  resultant gradient to a pre-allocated vector ``g``.

  When ``x`` is a matrix that comprises ``n`` samples (each being a column), then ``c`` must be a vector of
  length ``n``, it computes the linear combination of the gradients evaluated at the given samples, using the
  values in ``c`` as coefficients. Likewise, the resultant accumulated gradient is written to ``g``.

  ``g`` should be of the same size as ``theta``.

  **Note:** this function is mainly used by the internal of the optimization algorithms.


The package already provides several commonly used predictors:


Linear predictor
~~~~~~~~~~~~~~~~~

A *linear predictor* is a real-valued linear functional :math:`f: \mathbb{R}^d \rightarrow \mathbb{R}`, given by

.. math::

  f(x; \theta) := \theta^T x

In the package, a linear predictor is represented by the type ``LinearPredictor``:

.. code:: julia

  type LinearPredictor <: UnivariatePredictor
  end


Affine predictor
~~~~~~~~~~~~~~~~~

An *affine predictor* is a real-valued linear functional :math:`f: \mathbb{R}^d \rightarrow \mathbb{R}`, given by

.. math::

  f(x; \theta) := \theta_{1:d}^T x + \theta_{d+1} \cdot \mathrm{bias}

Note that the parameter :math:`\theta` is an ``d+1``-dimensional vector, which stacks the coefficients for features and a coefficient for the bias.

In the package, an affine predictor is represented by the type ``AffinePredictor``:

.. code:: julia

  type AffinePredictor{T<:FloatingPoint} <: UnivariatePredictor
      bias::T
  end

  AffinePredictor{T<:FloatingPoint}(bias::T) = AffinePredictor{T}(bias)
  AffinePredictor() = AffinePredictor(1.0)


Multivariate linear predictor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A *multivariate linear predictor* is a vector-valued linear functional :math:`f: \mathbb{R}^d \rightarrow \mathbb{R}^q`, given by

.. math::

  f(x; \theta) := \theta^T x

The parameter :math:`\theta` is a matrix of size ``(d, q)``.

In the package, a multivariate linear predictor is represented by the type ``MvLinearPredictor``:

.. code:: julia

  type MvLinearPredictor <: MultivariatePredictor
  end


Multivariate affine predictor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A *multivariate affine predictor* is a vector-valued linear functional :math:`f: \mathbb{R}^d \rightarrow \mathbb{R}^q`, given by

.. math::

  f_i(x; \theta) := \theta_{1:d, i}^T x + \theta_{d+1, i} \cdot \mathrm{bias}, \ \forall i = 1, \ldots, q

The parameter :math:`theta` is a matrix of size ``(d+1, q)``.

In the package, a multivariate affine predictor is represented by the type ``MvAffinePredictor``:

.. code:: julia

  type MvAffinePredictor{T<:FloatingPoint} <: MultivariatePredictor
      bias::T
  end

  MvAffinePredictor{T<:FloatingPoint}(bias::T) = MvAffinePredictor{T}(bias)
  MvAffinePredictor() = MvAffinePredictor(1.0)


**Note:** In the context of classification, one should *directly* use the value(s) yielded by the linear or affine predictors as arguments to the loss function (*e.g.* *logistic loss* or *multinomial logistic loss*), without converting them into class labels.
