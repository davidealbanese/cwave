# This code reproduces the figure 2a in (Torrence, 1998).
# The plot on the left give the real part (solid) and the imaginary part
# (dashed) for the Morlet wavelet in the time domain. The plot on the right
# give the corresponding wavelet in the frequency domain.
# Change the wavelet function in order to obtain the figures 2b 2c and 2d.

from __future__ import division

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import cwave


wavelet = cwave.Morlet() # Paul(), DOG() or DOG(m=6)
dt = 1
scale = 10*dt
t = np.arange(-40, 40, dt)
omega = np.arange(-1.2, 1.2, 0.01)

psi = wavelet.time(t, scale=scale, dt=dt)
psi_real = np.real(psi)
psi_imag = np.imag(psi)
psi_hat = wavelet.freq(omega, scale=scale, dt=dt)

fig = plt.figure(1)

ax1 = plt.subplot(121)
plt.plot(t/scale, psi_real, "k", label=("Real"))
plt.plot(t/scale, psi_imag, "k--", label=("Imag"))
plt.xlabel("t/s")
plt.legend()

ax2 = plt.subplot(122)
plt.plot((scale*omega)/(2*np.pi), psi_hat, "k")
plt.vlines(0, psi_hat.min(), psi_hat.max(), linestyles="dashed")
plt.xlabel("s * omega/(2 * PI)")

plt.show()
