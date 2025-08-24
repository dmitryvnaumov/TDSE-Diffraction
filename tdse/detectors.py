from __future__ import annotations
import numpy as np
from .units import hbar


def intensity_plane(psi: np.ndarray, grid: "Grid3D", z_det: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kz = grid.index_of_z(z_det)
    img = np.abs(psi[:, :, kz])**2
    return grid.x, grid.y, img


def flux_plane(psi: np.ndarray, grid: "Grid3D", z_det: float, mass: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kz = grid.index_of_z(z_det)
    # finite-diff dz derivative (central differences where possible)
    k0 = max(1, kz-1); k1 = min(grid.Nz-2, kz+1)
    dpsi_dz = (psi[:, :, k1] - psi[:, :, k0]) / ((k1 - k0)*grid.dz)
    jz = (hbar/mass) * np.imag(np.conj(psi[:, :, kz]) * dpsi_dz)
    return grid.x, grid.y, jz


def radial_average(x: np.ndarray, y: np.ndarray, img: np.ndarray, nbins: int = 200):
    X, Y = np.meshgrid(x, y, indexing='ij')
    R = np.sqrt(X*X + Y*Y).ravel()
    V = img.ravel()
    rmax = min(abs(x[0]), abs(x[-1]), abs(y[0]), abs(y[-1]))
    bins = np.linspace(0, rmax, nbins+1)
    which = np.digitize(R, bins) - 1
    prof = np.zeros(nbins)
    counts = np.zeros(nbins)
    for i, val in zip(which, V):
        if 0 <= i < nbins:
            prof[i] += val
            counts[i] += 1
    prof = np.divide(prof, counts, out=np.zeros_like(prof), where=counts>0)
    r = 0.5*(bins[:-1] + bins[1:])
    return r, prof
