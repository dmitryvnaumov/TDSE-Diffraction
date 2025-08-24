import numpy as np
from numpy.fft import fftn, ifftn

class ParaxialPropagator:
    def __init__(self, grid2d, k0: float, dz: float):
        self.g = grid2d
        self.k0 = float(k0)
        self.dz = float(dz)
        KX, KY = np.meshgrid(self.g.kx, self.g.ky, indexing='ij')
        self._U = np.exp(-1j * (KX**2 + KY**2) * self.dz / (2.0 * self.k0))

    def step(self, u_xy: np.ndarray) -> None:
        uk = fftn(u_xy)
        uk *= self._U
        u_xy[:] = ifftn(uk)

def apply_mask_2d(u_xy: np.ndarray, Axy: np.ndarray) -> None:
    if Axy.shape != u_xy.shape:
        raise ValueError("Mask shape must match field shape")
    u_xy *= Axy
