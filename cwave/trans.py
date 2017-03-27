##    Copyright 2017 cwave Developers <davide.albanese@gmail.com>

##    This file is part of cwave.
##
##    cwave is free software: you can redistribute it and/or modify
##    it under the terms of the GNU General Public License as published by
##    the Free Software Foundation, either version 3 of the License, or
##    (at your option) any later version.
##
##    cwave is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.

##    You should have received a copy of the GNU General Public License
##    along with cwave. If not, see <http://www.gnu.org/licenses/>.

from __future__ import division

import copy

import numpy as np
import scipy as sp

from . import wavefun


class Series(object):
    """Series.

    Example
    -------
    >>> import cwave
    >>> import numpy as np
    >>> x = np.random.sample(16)
    >>> dt=2
    >>> s = cwave.Series(x, dt)
    """

    def __init__(self, x, dt=1):
        """
        Parameters
        ----------
        x : 1d array_like object
            series (N).
        dt : float
            time step.
        """

        x_arr = np.array(x, dtype=np.float, copy=True)
        if x_arr.ndim != 1:
            raise ValueError('x must be an 1d array_like object')
        
        if dt <= 0.0:
            raise ValueError('dt must be > 0')
     
        self._x = x_arr
        self._dt = dt

    @property
    def x(self):
        """Get the series (1d numpy array).
        """
        
        return self._x
    
    @property
    def dt(self):
        """Get the time step (float).
        """

        return self._dt


class Trans(object):
    """Continuous wavelet transformed series.
    """

    def __init__(self, W, scales, omega, wavelet, x_mean=0, dt=1):
        """
        Parameters
        ----------
        W : 2d array_like object
            transformed series (J x N).
        scales : 1d array_like object
            wavelet scales (J).
        omega : 1d array_like object
            angular frequencies (N).
        wavelet : Wavelet object
            wavelet used for transformation.
        x_mean : float
            mean value of the original (non-transformed) series.
        dt : float
            time step.        
        """

        W_arr = np.array(W, copy=True)
        if W_arr.ndim != 2:
            raise ValueError('W must be an 2d array_like object')

        scales_arr = np.array(scales, dtype=np.float, copy=True)
        if scales_arr.ndim != 1:
            raise ValueError('scales must be an 1d array_like object')

        omega_arr = np.array(omega, dtype=np.float, copy=True)
        if omega_arr.ndim != 1:
            raise ValueError('omega must be an 1d array_like object')

        if (W_arr.shape[0] != scales_arr.shape[0]) or \
            (W_arr.shape[1] != omega_arr.shape[0]):
            raise ValueError('shape of W must be (n_scales, n_omega)')
        
        if not isinstance(wavelet, wavefun.Wavelet):
            raise ValueError('wavelet must be an Wavelet object')

        if dt <= 0.0:
            raise ValueError('dt must be > 0')

        self._W = W_arr
        self._scales = scales_arr
        self._omega = omega_arr
        self._wavelet = copy.deepcopy(wavelet)
        self._x_mean = x_mean
        self._dt = dt

    @property
    def W(self):
        """Get the transformed series (2d numpy array).
        """
        
        return self._W
    
    @property
    def scales(self):
        """Get the wavelet scales (1d numpy array).
        """

        return self._scales

    @property
    def omega(self):
        """Get the angular frequencies (1d numpy array).
        """

        return self._omega

    @property
    def wavelet(self):
        """Get the wavelet (Wavelet).
        """

        return self._wavelet

    @property
    def x_mean(self):
        """Get the mean value of the original (non-transformed) series (float).
        """

        return self._x_mean

    @property
    def dt(self):
        """Get the time step (float).
        """

        return self._dt

    def S(self):
        """Returns the wavelet power spectrum abs(W)^2."""

        return np.absolute(self._W)**2

    def var(self):
        """Returns the variance (eq. 14 in Torrence 1998)"""

        # (dj * dt) / Cd = psi0(0) * sqrt(dt) * Ck

        Ck = self._wavelet._factor_Ck(self._scales, self._omega, self._dt)
        psi00 = self._wavelet.time(0)

        return ((psi00 * np.sqrt(self._dt) * Ck) / self._W.shape[1]) * \
            np.sum(self.S() / self._scales.reshape(-1, 1)) 


def cwt(s, wavelet, dj=0.25, scale0=None, scales=None):
    """Continuous wavelet transform.
    
    Parameters
    ----------
    s : :class:`cwave.Series` object
        series.
    wavelet : :class:`cwave.Wavelet` object
        wavelet function.
    dj : float
        scale resolution (spacing between scales). A smaller dj will give better
        scale resolution. If scales is not None, this parameter is ignored.
    scale0 : float
        the smallest scale. If scale0=None it is chosen so that the 
        equivalent Fourier period is 2*dt. If scales is not None, this parameter
        is ignored.
    scales : None, float or 1d array_like object
        wavelet scale(s). If scales=None, the scales are automatically computed
        as fractional powers of two, with a scale resolution dj and the smallest
        scale scale0. If the parameter scales is provided, dj and scale0 are
        automatically ignored.

    Returns
    -------
    T : :class:`cwave.Trans` object
        continuous wavelet transformed series.

    Example
    -------
    >>> import cwave
    >>> import numpy as np
    >>> x = np.random.sample(8)
    >>> dt=2
    >>> T = cwave.cwt(cwave.Series(x, dt), cwave.DOG(), scales=[2, 4, 8])
    >>> T.S()
    array([[  1.19017195e-01,   1.54253543e-01,   9.07432163e-02,
              2.96227293e-02,   5.89519189e-04,   1.09486971e-01,
              1.15546678e-02,   3.93108223e-02],
           [  4.43627788e-01,   2.27266757e-01,   3.35649411e-05,
              1.86687786e-01,   3.78349904e-01,   2.24894861e-01,
              3.22011576e-03,   1.84538790e-01],
           [  8.01208492e-03,   4.40859411e-03,   1.92688516e-05,
              3.62275119e-03,   8.01206572e-03,   4.40859342e-03,
              1.92697929e-05,   3.62275056e-03]])
    """
    
    if not isinstance(wavelet, wavefun.Wavelet):
        raise ValueError('wavelet must be an Wavelet object')

    N = s.x.shape[0]
    dt = s.dt

    if scales is not None:
        scales_arr = np.atleast_1d(scales)
    else:
        scales_arr = wavelet.auto_scales(dt, dj, N, scale0)

    x_mean = np.mean(s.x)
    x_centered = s.x - x_mean
    
    omega = 2 * np.pi * np.fft.fftfreq(N, dt)
    x_hat = np.fft.fft(x_centered)

    W = []
    for scale in scales_arr:
        wavelet_hat = wavelet.freq(omega, scale, dt)
        W.append(np.fft.ifft(wavelet_hat * x_hat))
    
    return Trans(W, scales_arr, omega, wavelet, x_mean, dt)


def icwt(T):
    """Inverse continuous wavelet transform.

    Parameters
    ----------
    T : :class:`cwave.Trans` object
        continuous wavelet transformed series.

    Returns
    -------
    s : :class:`cwave.Series` object
        series.

    Example
    -------
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
    """

    if not isinstance(T, Trans):
        raise ValueError('W must be a cwave.Trans object')

    Ck = T.wavelet._factor_Ck(T.scales, T.omega, T.dt)
    x = Ck * np.sum(np.real(T.W) / np.sqrt(T.scales).reshape(-1, 1), axis=0)

    return Series(x + T.x_mean, T.dt)
    