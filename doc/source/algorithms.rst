Optimization Algorithms
========================

The package provides functions that implement SGD and its variants.

.. function:: sgd(pred, loss, theta, stream[, ...])

    Standard Stochastic Gradient Descent.

    :param pred: The predictor.
    :param loss: The loss function.
    :param theta: The initial guess of the solution.
    :param stream: The data stream.

    :return: The resultant solution.

    This function also supports the following keyword arguments.

    ================ ====================== ===================================================
     name             default                 description
    ================ ====================== ===================================================
     ``reg``          ``NoReg()``            The regularizer.

     ``lrate``        ``t->1.0/(1.0 + t)``   The rule of learning rate, which should be
                                             a function of the iteration number ``t``.

     ``cbinterval``   ``0``                  The interval of invoking callback.

                                             - ``0``: never invoke callback.
                                             - ``1``: invoke callback at each iteration.
                                             - ``k``: invoke callback every ``k`` iterations.
                                             
     ``callback``     ``simple_trace``       The callback function.
                                             (See :ref:`callback` for details).
    ================ ====================== ===================================================

    **Note:** The number of iterations is determined by the number of items in the data stream. One can change the behavior of the algorithm by constructing the data stream in different ways. (See :ref:`stream` for details)


*More algorithms are being implemented. We will document these algorithms as we proceed.*
