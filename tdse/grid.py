from __future__ import annotations
import numpy as np
from numpy.fft import fftfreq
from dataclasses import dataclass

@dataclass(frozen=True)
class Grid3D:
    Nx: int; Ny: int; Nz: int
    dx: float; dy: float; dz: float  # meters

    def __post_init__(self):
        if min(self.Nx, self.Ny, self.Nz) < 4:
            raise ValueError("Grid must have at least 4 points in each dimension.")
        if min(self.dx, self.dy, self.dz) <= 0:
            raise ValueError("Spacings must be positive.")

    # Axes (cached properties)
    @property
    def x(self) -> np.ndarray:
        return (np.arange(self.Nx) - self.Nx//2) * self.dx

    @property
    def y(self) -> np.ndarray:
        return (np.arange(self.Ny) - self.Ny//2) * self.dy

    @property
    def z(self) -> np.ndarray:
        return (np.arange(self.Nz) - self.Nz//2) * self.dz

    # Wavenumber axes (rad/m)
    @property
    def kx(self) -> np.ndarray:
        return 2*np.pi*fftfreq(self.Nx, d=self.dx)

    @property
    def ky(self) -> np.ndarray:
        return 2*np.pi*fftfreq(self.Ny, d=self.dy)

    @property
    def kz(self) -> np.ndarray:
        return 2*np.pi*fftfreq(self.Nz, d=self.dz)

    # Helpers
    @property
    def Lx(self) -> float: return self.Nx*self.dx
    @property
    def Ly(self) -> float: return self.Ny*self.dy
    @property
    def Lz(self) -> float: return self.Nz*self.dz

    def index_of_z(self, z0: float) -> int:
        """Nearest index to physical z0 (m)."""
        return int(np.argmin(np.abs(self.z - z0)))

    def memory_bytes(self, complex_arrays: int = 2) -> int:
        # complex128 = 16 bytes per element
        return self.Nx*self.Ny*self.Nz * 16 * complex_arrays
