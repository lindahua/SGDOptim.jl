.. _stream:

Sample Streams
================

Unlike conventional methods, SGD and its variants look at a single sample or a small batch of samples at each iteration. In other words, data are viewed as a stream of samples or minibatches.

This package provides a variety of ways to construct data streams. Each data stream is essentially an iterator that implements the ``start``, ``done``, and ``next`` methods (see `here <http://julia.readthedocs.org/en/latest/stdlib/collections/#iteration>`_ for details of Julia's iteration patterns). Each item from a data stream can be either a sample (as a pair of input and output) or a mini-batch (as a pair of multi-input array and multi-output array).

**Note:** All SGD algorithms in this package support both sample streams and mini-batch streams. At each iteration, the algorithm works on a single item from the stream, which can be either a sample or a mini-batch.


Wrap Arrays into Data Streams
-------------------------------

The package provides several methods to construct streams of samples or minibatches.

.. function:: sample_seq(X, Y[, ord])

    Wrap an input array ``X`` and an output array ``Y`` into a stream of individual samples.

    Each item of the stream is a pair, comprised of an item from ``X`` and a corresponding item from ``Y``. If ``X`` is a vector, then each item of ``X`` is a scalar, if ``X`` is a matrix, then each item of ``X`` is a column vector. The same applies to ``Y``.

    The ``ord`` argument is an instance of ``AbstractVector`` that specifies the order in which the samples are scanned. If ``ord`` is omitted, it is, by default, set to the natural order, namely,
    ``1:n``, where ``n`` is the number of samples in the data set.

.. function:: minibatch_seq(X, Y, bsize[, ord])

    Wrap an input array ``X`` and an output array ``Y`` into a stream of mini-batches of size ``bsize`` or smaller.

    For example, if ``X`` and ``Y`` have ``28`` samples, by setting ``bsize`` to ``10``, we partition the data set into three minibatches, respectively corresponding to the indices ``1:10``, ``11:20``, and ``21:28``.

    The ``ord`` argument specifies the order in which the mini-batches are used. For example, if ``ord`` is set to ``[3, 2, 1]``, it first takes the 3rd batch, then 2nd, and finally 1st. If ``ord`` is omitted, it is, by default, set to the natural order, namely, ``1:m``, where ``m`` is the number of mini-batches.
