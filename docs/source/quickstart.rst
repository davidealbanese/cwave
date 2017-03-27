.. currentmodule:: cwave

Quickstart
==========

Install
-------

The easiest way to install :mod:`cwave` is using pip:

.. code-block:: sh

    $ pip install cwave

In Mac OS X/MacOS, we recommend to install Python from
`Homebrew <http://brew.sh/>`_.

You can also install :mod:`cwave` from source (with the command
``python setup.py install``).


Using :mod:`cwave`
------------------

.. code-block:: sh

    $ python

.. code-block:: python

    >>> import cwave
    >>> import numpy as np
    >>> x = np.random.normal(2, 1, 16)
    >>> x
    array([ 1.6960901 ,  3.27146653,  2.55896222,  2.39484518,  2.34977766,
            3.48575552,  1.7372688 , -0.21329766,  3.5425618 ,  3.34657898,
            1.54359934,  2.96181542,  2.43294205,  1.65980233,  3.44710306,
            1.69615204])
    >>> dt=1
    >>> T = cwave.cwt(cwave.Series(x, dt), cwave.DOG())
    >>> sr = cwave.icwt(T)
    >>> sr.x
    array([ 1.64067578,  3.28517018,  2.78434897,  2.38949828,  2.58014315,
            3.52751356,  1.34776275, -0.41078628,  3.3648406 ,  3.56461166,
            1.7286081 ,  2.88596331,  2.40275636,  1.81964648,  3.28932397,
            1.7113465 ])