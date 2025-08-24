from dataclasses import dataclass
import numpy as np
from numpy.fft import fftfreq

@dataclass(frozen=True)
class Grid2D:
    Nx: int; Ny: int
    dx: float; dy: float  # meters

    @property
    def x(self): return (np.arange(self.Nx) - self.Nx//2) * self.dx
    @property
    def y(self): return (np.arange(self.Ny) - self.Ny//2) * self.dy
    @property
    def kx(self): return 2*np.pi*fftfreq(self.Nx, d=self.dx)
    @property
    def ky(self): return 2*np.pi*fftfreq(self.Ny, d=self.dy)
    @property
    def Lx(self): return self.Nx*self.dx
    @property
    def Ly(self): return self.Ny*self.dy
