from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .units import hbar, m_e

@dataclass
class State:
    grid: "Grid3D"
    psi: np.ndarray  # complex128 shape (Nx,Ny,Nz)

    @classmethod
    def empty(cls, grid: "Grid3D") -> "State":
        return cls(grid, np.zeros((grid.Nx, grid.Ny, grid.Nz), dtype=np.complex128))

    def normalize(self) -> float:
        dx,dy,dz = self.grid.dx, self.grid.dy, self.grid.dz
        n = (np.vdot(self.psi, self.psi).real) * dx*dy*dz
        if n > 0:
            self.psi /= np.sqrt(n)
        return n

    def plane(self, k_z: int) -> np.ndarray:
        return self.psi[:, :, k_z]

# --- Initial conditions ---

def make_gaussian_beam(grid: "Grid3D", E_eV: float, waist: float,
                        z0: float, kdir: tuple[float,float,float] = (0,0,1)) -> State:
    from .units import k0_from_energy
    X, Y, Z = np.meshgrid(grid.x, grid.y, grid.z, indexing='ij')
    k0 = k0_from_energy(E_eV)
    kvec = k0 * np.array(kdir)/np.linalg.norm(kdir)
    rho2 = X**2 + Y**2
    psi = np.exp(-rho2/(waist**2)) * np.exp(1j*(kvec[0]*X + kvec[1]*Y + kvec[2]*(Z - z0)))
    st = State(grid, psi.astype(np.complex128))
    st.normalize()
    return st

def make_vortex_beam(grid: "Grid3D", E_eV: float, waist: float, z0: float, ell: int = 1,
                     kdir: tuple[float,float,float] = (0,0,1)) -> State:
    from .units import k0_from_energy
    X, Y, Z = np.meshgrid(grid.x, grid.y, grid.z, indexing='ij')
    k0 = k0_from_energy(E_eV)
    kvec = k0 * np.array(kdir)/np.linalg.norm(kdir)
    phi = np.arctan2(Y, X)
    rho2 = X**2 + Y**2
    transverse = (np.sqrt(rho2)/waist)**abs(ell) * np.exp(-rho2/(waist**2)) * np.exp(1j*ell*phi)
    psi = transverse * np.exp(1j*(kvec[0]*X + kvec[1]*Y + kvec[2]*(Z - z0)))
    st = State(grid, psi.astype(np.complex128))
    st.normalize()
    return st
