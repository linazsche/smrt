"""
Contains different solvers of the radiative transfer equation. Based on the electromagnetic properties of
each layer computed by the EM model, these RT solvers compute the emission and propagation of energy in the medium up to the surface (the atmosphere is usually
dealt with independently in dedicated modules in :py:mod:`smrt.atmosphere`).

The solvers differ by the approximations and numerical methods. :py:mod:`~smrt.rtsolver.dort` is currently the most accurate and recommended
in most cases unless the computation time is a constraint.

Selection of the solver is done with the :py:mod:`smrt.core.model.make_model` function.

Note:  **For Developers**

    To experiment with DORT, it is recommended to copy the file dort.py to e.g. dort_mytest.py so it is immediately available through
    :py:mod:`smrt.core.model.make_model`.

    To develop a new solver that will be accessible by the :py:mod:`smrt.core.model.make_model` function, add
    a file in this directory. Refer to dort.py as an example. Only the `solve` method needs
    to be implemented. It must return a :py:mod:`smrt.core.result.Result` instance with the results. Contact the core developers for more details.
"""
