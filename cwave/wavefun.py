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

import abc

import six

import numpy as np
import scipy as sp
import scipy.misc
import scipy.special


@six.add_metaclass(abc.ABCMeta)
class Wavelet(object):
    """The abstract wavelet base class.
    """
    
    @abc.abstractmethod
    def time(self, t, scale=1, dt=None):
        """Wavelet function in the time domain psi0(t/s).

        Parameters
        ----------
        t : float
            time. If df is not None each element of t should be multiple of dt.
        scale : float
            wavelet scale.
        dt : None or float
            time step. If dt is not None, the method returns the 
            energy-normalized psi (psi0). 

        Returns
        -------
        psi0 : float/complex or 1D numpy array
            the wavelet in the time domain.

        Example
        -------
        >>> import cwave
        >>> w = cwave.Morlet()
        >>> t = range(-8, 8)
        >>> w.time(t, 10)
        array([ 0.04772449+0.54333716j, -0.28822893+0.51240757j,
               -0.56261977+0.27763414j, -0.65623233-0.09354365j,
               -0.51129130-0.46835014j, -0.16314795-0.69929479j,
                0.26678672-0.68621589j,  0.61683875-0.42200209j,
                0.75112554+0.j        ,  0.61683875+0.42200209j,
                0.26678672+0.68621589j, -0.16314795+0.69929479j,
               -0.51129130+0.46835014j, -0.65623233+0.09354365j,
               -0.56261977-0.27763414j, -0.28822893-0.51240757j])
        """

    @abc.abstractmethod
    def freq(self, omega, scale=1, dt=None):
        """Wavelet function in the frequency domain psi0_hat(s*omega).

        Parameters
        ----------
        omega : float or 1d array_like object
            angular frequency (length N).
        scale : float
            wavelet scale.
        dt : None or float
            time step. If dt is not None, the method returns the 
            energy-normalized psi_hat (psi_hat0).

        Returns
        -------
        psi0_hat : float/complex or 1D numpy array
            the wavelet in the frequency domain

        Example
        -------
        >>> import numpy as np
        >>> import cwave
        >>> w = cwave.Morlet()
        >>> omega = 2 * np.pi * np.fft.fftfreq(32, 1)
        >>> w.freq(omega, 10)
        array([  0.00000000e+000,   2.17596717e-004,   8.76094852e-002,
                 7.46634798e-001,   1.34686366e-001,   5.14277294e-004,
                 4.15651521e-008,   7.11082035e-014,   2.57494732e-021,
                 1.97367345e-030,   3.20214178e-041,   1.09967442e-053,
                 7.99366604e-068,   1.22994640e-083,   4.00575772e-101,
                 2.76147731e-120,   0.00000000e+000,   0.00000000e+000,
                 0.00000000e+000,   0.00000000e+000,   0.00000000e+000,
                 0.00000000e+000,   0.00000000e+000,   0.00000000e+000,
                 0.00000000e+000,   0.00000000e+000,   0.00000000e+000,
                 0.00000000e+000,   0.00000000e+000,   0.00000000e+000,
                 0.00000000e+000,   0.00000000e+000])
        """

    @abc.abstractmethod
    def fourier_period(self, scale):
        """Computes the equivalent Fourier period (wavelength) from wavelet
        scale.

        Parameters
        ----------
        scale : float
            wavelet scale.

        Returns
        -------
        lmbd : float
            equivalent Fourier period.

        Example
        -------
        >>> import cwave
        >>> w = cwave.Morlet()
        >>> w.fourier_period(scale=1)
        1.0330436477492537
        """

    def wavelet_scale(self, lmbd):
        """Computes the wavelet scale from equivalent Fourier period
        (wavelength).

        Parameters
        ----------
        lmbd : float
            equivalent Fourier period.

        Returns
        -------
        scale : float
            wavelet scale.

        Example
        -------
        >>> import cwave
        >>> w = cwave.Morlet()
        >>> w.wavelet_scale(lmbd=10)
        9.6801330919439117
        """
      
        scale = self.fourier_period(1)**(-1) * lmbd

        return scale

    @abc.abstractmethod
    def efolding_time(self, scale):
        """Returns the e-folding time tau_s.

        Parameters
        ----------
        scale : float
            wavelet scale.

        Returns
        -------
        tau : float
            e-folding time.

        Example
        -------
        >>> import cwave
        >>> w = cwave.Morlet()
        >>> w.efolding_time(1)
        1.4142135623730951
        """

    def smallest_scale(self, dt):
        """Returns the smallest resolvable scale. It is chosen so that the
        equivalent Fourier period is 2*dt.

        Parameters
        ----------
        dt : float
            time step.

        Returns
        -------
        scale0 : float
            smallest scale.

        Example
        -------
        >>> import cwave
        >>> w = cwave.Morlet()
        >>> w.smallest_scale(dt=1)
        1.9360266183887822
        """

        scale0 = self.wavelet_scale(2*dt)

        return scale0


    def auto_scales(self, dt, dj, N, scale0=None):
        """Computes the equivalent Fourier period (wavelength) from wavelet
        scale.

        Parameters
        ----------
        dt : float
            time step.
        dj : float
            scale resolution. For the Morlet wavelet, a dj of about 0.5 is the
            largest value that still gives adequate sampling in scale, while for
            the other wavelet functions al larger value can be used.
        N : integer
            number of data samples.
        scale0 : float
            the smallest scale. If scale0=None it is chosen so that the 
            equivalent Fourier period is 2*dt.

        Returns
        -------
        scale : 1d numpy array
            scales.

        Example
        -------
        >>> import cwave
        >>> w = cwave.Morlet()
        >>> w.auto_scales(dt=1, dj=0.125, N=64, scale0=1)
        array([  1.        ,   1.09050773,   1.18920712,   1.29683955,
                 1.41421356,   1.54221083,   1.68179283,   1.83400809,
                 2.        ,   2.18101547,   2.37841423,   2.59367911,
                 2.82842712,   3.08442165,   3.36358566,   3.66801617,
                 4.        ,   4.36203093,   4.75682846,   5.18735822,
                 5.65685425,   6.1688433 ,   6.72717132,   7.33603235,
                 8.        ,   8.72406186,   9.51365692,  10.37471644,
                 11.3137085 ,  12.3376866 ,  13.45434264,  14.67206469,
                 16.        ,  17.44812372,  19.02731384,  20.74943287,
                 22.627417  ,  24.67537321,  26.90868529,  29.34412938,
                 32.        ,  34.89624745,  38.05462768,  41.49886575,
                 45.254834  ,  49.35074641,  53.81737058,  58.68825877,  64.        ])
        """

        if scale0 is None:
            scale0 = self.smallest_scale(dt)
            
        J = int((1 / dj) * np.log2((N * dt) / scale0))
        scale = scale0 * 2**(dj * np.arange(J+1))

        return scale

    def _factor_Ck(self, scales, omega, dt):
        """The reconstruction factor Ck is defined as 
        (dj*sqrt(dt))/(Cd*psi0(0)) = 1/(sum(real(Wd(sj))/sqrt(sj))
        (see eq. 13 in Torrence 1998).

        Parameters
        ----------
        omega : float or 1d array_like object
            angular frequency(es).
        scale : float or 1d array_like object
            wavelet scale(s).
        dt : float
            time step.

        Returns
        -------
        Ck : 1d numpy array
            reconstruction factor Ck.
        """

        scales_arr = np.atleast_1d(scales)
        if scales_arr.ndim > 1:
            raise ValueError("scales must be float or an 1d array_like object")

        omega_arr = np.atleast_1d(omega)
        if omega_arr.ndim > 1:
            raise ValueError("omega must be float or an 1d array_like object")

        if dt <= 0:
            raise ValueError('dt must be > 0')

        N = omega_arr.shape[0]
        
        d = 0.
        for scale in scales_arr:
            Wd = self.freq(omega_arr, scale, dt).sum() / N
            d += np.sum(np.real(Wd) / np.sqrt(scale))
        Ck = 1 / d

        return Ck

    def _factor_Cd(self, scales, omega, dt, dj):
        """The factor Cd is defined in eq. 13 in (Torrence 1998).

        Parameters
        ----------
        omega : float or 1d array_like object
            angular frequency(es).
        scale : float or 1d array_like object
            wavelet scale(s).
        dt : integer
            number of data samples.
        dj : float
            scale resolution.

        Returns
        -------
        Ck : 1d numpy array
            reconstruction factor Ck.
        """

        if dj <= 0:
            raise ValueError('dj must be > 0')

        Ck = self._factor_Ck(scales, omega, dt)
        Cd = (dj * np.sqrt(dt)) / (Ck * self.time(0))

        return Cd

    def _norm_freq(self, scale, dt):
        return np.sqrt((2 * np.pi * scale) / dt)

    def _norm_time(self, scale, dt):
        return np.sqrt(dt / scale)


class Morlet(Wavelet):
    """Morlet wavelet function.

    Example
    -------
    >>> import cwave
    >>> w = cwave.Morlet(omega0=6)
    """
    
    def __init__(self, omega0=6):
        """
        Parameters
        ----------
        omega0 : float
            nondimesional frequency. It should be >=6 in order to satisfy
            the admissibility condition (see Torrence 1998, Farge 1992).
        """
        
        self._omega0 = omega0

    @property
    def omega0(self):
        """Get the omega0 parameter"""

        return self._omega0

    def time(self, t, scale=1, dt=None):

        t_arr = np.asarray(t)
        if t_arr.ndim > 1:
            raise ValueError("t must be a float or an 1d array_like object")

        eta = t_arr / scale

        psi = np.pi**(-0.25) * np.exp(1.j * self._omega0 * eta) * \
            np.exp(-0.5 * (eta**2))

        if dt is not None:
            psi *= self._norm_time(scale, dt)

        return psi
    time.__doc__ = Wavelet.time.__doc__

    def freq(self, omega, scale=1, dt=None):

        omega_arr = np.asarray(omega)
        if omega_arr.ndim > 1:
            raise ValueError("omega must be a float or an 1d array_like object")

        psi_hat = np.zeros(omega_arr.shape[0], dtype=np.float)

        H = (omega_arr > 0)
        k = omega_arr[H] * scale
        psi_hat[H] = np.pi**(-0.25) * np.exp(-(k - self._omega0)**2 / 2.)

        if dt is not None:
            psi_hat *= self._norm_freq(scale, dt)

        return psi_hat
    freq.__doc__ = Wavelet.freq.__doc__

    def fourier_period(self, scale):

        lmbd = (4 * np.pi * scale) / \
            (self._omega0 + np.sqrt(2 + self._omega0**2))

        return lmbd
    fourier_period.__doc__ = Wavelet.fourier_period.__doc__

    def efolding_time(self, scale):

        tau = np.sqrt(2) * scale

        return tau
    efolding_time.__doc__ = Wavelet.efolding_time.__doc__


class Paul(Wavelet):
    """Paul Wavelet function.

    Example
    -------
    >>> import cwave
    >>> w = cwave.Paul(m=4)
    """

    def __init__(self, m=4):
        """
        Parameters
        ----------
        m : float
            order.
        """
        
        self._m = m

    @property
    def m(self):
        """Get the m parameter (order)"""

        return self._m

    def time(self, t, scale=1, dt=None):

        t_arr = np.asarray(t)
        if t_arr.ndim > 1:
            raise ValueError("t must be a float or an 1d array_like object")

        eta = t_arr / scale

        psi = (1- 1j*eta)**(-(self._m+1)) * \
            (2**self._m * 1j**self._m * scipy.misc.factorial(self._m)) / \
            np.sqrt(np.pi * scipy.misc.factorial(2*self._m))

        if dt is not None:
            psi *= self._norm_time(scale, dt)

        return psi
    time.__doc__ = Wavelet.time.__doc__

    def freq(self, omega, scale=1, dt=None):

        omega_arr = np.asarray(omega)
        if omega_arr.ndim > 1:
            raise ValueError("omega must be a float or an 1d array_like object")

        psi_hat = np.zeros(omega_arr.shape[0], dtype=np.float)

        H = (omega_arr > 0)
        k = omega_arr[H] * scale
        psi_hat[H] = k**self._m * np.exp(-k) * 2**self._m / \
            np.sqrt(self._m * scipy.misc.factorial(2 * self._m - 1))

        if dt is not None:
            psi_hat *= self._norm_freq(scale, dt)

        return psi_hat
    freq.__doc__ = Wavelet.freq.__doc__

    def fourier_period(self, scale):

        lmbd = (4 * np.pi * scale) / (2 * self._m + 1)

        return lmbd
    fourier_period.__doc__ = Wavelet.fourier_period.__doc__

    def efolding_time(self, scale):

        tau = scale / np.sqrt(2)

        return tau
    efolding_time.__doc__ = Wavelet.efolding_time.__doc__


class DOG(Wavelet):
    """Derivative Of Gaussian (DOG) Wavelet function.

    Example
    -------
    >>> import cwave
    >>> w = cwave.DOG(m=2)
    """

    def __init__(self, m=2):
        """
        Parameters
        ----------
        m : float
            derivative.
        """
        
        self._m = m

    @property
    def m(self):
        """Get the m parameter (derivative)"""

        return self._m

    def time(self, t, scale=1, dt=None):

        t_arr = np.asarray(t)
        if t_arr.ndim > 1:
            raise ValueError("t must be a float or an 1d array_like object")

        eta = t_arr / scale

        hn = scipy.special.hermitenorm(self._m)
        psi = -(hn(eta) * np.exp(-eta**2/2)) / \
            np.sqrt(scipy.special.gamma(self._m+.5))

        if dt is not None:
            psi *= self._norm_time(scale, dt)

        return psi
    time.__doc__ = Wavelet.time.__doc__

    def freq(self, omega, scale=1, dt=None):

        omega_arr = np.asarray(omega)
        if omega_arr.ndim > 1:
            raise ValueError("omega must be a float or an 1d array_like object")

        k = omega_arr * scale
        
        psi_hat = k**self._m * np.exp(-k**2 / 2) / \
            np.sqrt(scipy.special.gamma(self._m+.5))
        
        if dt is not None:
            psi_hat *= self._norm_freq(scale, dt)

        return psi_hat
    freq.__doc__ = Wavelet.freq.__doc__

    def fourier_period(self, scale):

        lmbd = (2 * np.pi * scale) / np.sqrt(self._m + .5)

        return lmbd
    fourier_period.__doc__ = Wavelet.fourier_period.__doc__

    def efolding_time(self, scale):

        tau = np.sqrt(2) * scale

        return tau
    efolding_time.__doc__ = Wavelet.efolding_time.__doc__
