__version__ = "1.0"

from .trans import Series, Trans, cwt, icwt
from .wavefun import Wavelet, Morlet, Paul, DOG

__all__ = ["Series", "Trans", "cwt", "icwt", "Wavelet", "Morlet", "Paul", "DOG"]
