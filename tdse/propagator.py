from __future__ import annotations
import numpy as np
from numpy.fft import fftn, ifftn
from dataclasses import dataclass
from .units import hbar, m_e

@dataclass
class SplitStep:
    grid: "Grid3D"
    mass: float
    dt: float

    def __post_init__(self):
        kx, ky, kz = self.grid.kx, self.grid.ky, self.grid.kz
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        K2 = KX*KX + KY*KY + KZ*KZ
        self._T_phase = np.exp(-1j * (hbar*self.dt/(2*self.mass)) * K2, dtype=np.complex128)

    def step(self, psi: np.ndarray, V_half: np.ndarray | None = None,
             boundary_mask: np.ndarray | None = None) -> None:
        """In-place Strang step. V_half and boundary_mask are multiplicative factors.
        V_half can be 1 or array same shape as psi.
        """
        if V_half is not None:
            psi *= V_half
        # kinetic
        psi_k = fftn(psi)
        psi_k *= self._T_phase
        psi[:] = ifftn(psi_k)
        # potential half
        if V_half is not None:
            psi *= V_half
        # boundary mask at the very end
        if boundary_mask is not None:
            psi *= boundary_mask
