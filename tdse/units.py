"""Physical constants (SI) and helpers for electron beams.
No external deps.
"""
from __future__ import annotations
import numpy as np

# CODATA-like constants (SI)
h = 6.62607015e-34        # Planck constant (exact by definition)
hbar = h/(2*np.pi)
q_e = 1.602176634e-19     # elementary charge (C)
m_e = 9.1093837015e-31    # electron mass (kg)
c = 299792458.0           # speed of light (m/s, exact)

# --- Energy ↔ wavelength for electrons ---

def energy_eV_to_J(E_eV: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(E_eV) * q_e

def electron_wavelength(E_eV: float | np.ndarray) -> float | np.ndarray:
    """Relativistic de Broglie wavelength (m) for kinetic energy E (eV).
    Uses p = sqrt(E(E+2 m c^2)) / c and lambda = h/p.
    """
    E = energy_eV_to_J(E_eV)
    mc2 = m_e*c**2
    p = np.sqrt(E*(E + 2*mc2)) / c
    lam = h / p
    return lam

def k0_from_energy(E_eV: float | np.ndarray) -> float | np.ndarray:
    """Central wavenumber (rad/m) from kinetic energy (eV)."""
    lam = electron_wavelength(E_eV)
    return 2*np.pi/lam

# --- Heuristics for dt ---

def suggest_dt(dx: float, dy: float, dz: float, mass: float = m_e,
               phase_limit: float = 0.3*np.pi) -> float:
    """Suggest a time step (s) so that max kinetic phase per step is below phase_limit.
    Uses k_max ≈ π/min(dx,dy,dz).
    """
    dmin = min(dx, dy, dz)
    kmax = np.pi/dmin
    # phi_T = (ħ Δt / (2m)) kmax^2 <= phase_limit
    dt = phase_limit * 2*mass/(hbar * kmax*kmax)
    return dt
