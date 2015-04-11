.. _callback:

Callbacks
===========

The algorithms provided in this package interoperate with the rest of the world through *callbacks*. In particular, it allows a third party (*e.g.* a higher-level script, a user, a GUI, etc) to monitor the progress of the optimization and take proper actions.

Generally, a *callback* is an arbitrary function (or closure) that can be called in the following way:

.. function:: callback(theta, t, n, v)

    :param theta: The current solution.
    :param t:     The number of elapsed iterations.
    :param n:     The number of samples that have been used.
    :param v:     The objective value of the last item, which can be an objective evaluated on a single             sample or the total objective value evaluated on the last batch of samples.

The package already provides some callbacks for simple use:

.. function:: simple_trace

    Simply print the optimization trace, including the number of iterations, and the average loss of the last iteration.

    This is the default choice for most algorithms.


.. function:: gtcompare_trace(theta_g)

    In addition to printing the optimization trace, it also computes and shows the deviation from a given oracle ``theta_g``.

    **Note:** ``gtcompare_trace`` is a high-level function, and ``gtcompare_trace(theta_g)`` produces a callback function.
