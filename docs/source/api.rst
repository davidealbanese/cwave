.. currentmodule:: cwave

API
===

Summary
-------

Series and transformed data
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    Series
    Trans

Functions
^^^^^^^^^

.. autosummary::
    cwt
    icwt

Wavelet functions
^^^^^^^^^^^^^^^^^

The The :class:`Wavelet` abstract class
"""""""""""""""""""""""""""""""""""""""

All the wavelet function classes inherit from the abstract class
:class:`Wavelet`. The :class:`Wavelet` has the following abstract methods:

.. autosummary::
    Wavelet.time
    Wavelet.freq
    Wavelet.fourier_period
    Wavelet.efolding_time

Moreover, the :class:`Wavelet` exposes the following methods (available to the 
subclasses):

.. autosummary::
    Wavelet.wavelet_scale
    Wavelet.efolding_time
    Wavelet.smallest_scale
    Wavelet.auto_scales

Available wavelet classes
"""""""""""""""""""""""""

.. autosummary::
    Morlet
    Paul
    DOG

Classes and functions
---------------------

.. automodule:: cwave
    :members:
    :undoc-members:
