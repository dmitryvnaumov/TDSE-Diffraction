# TDSE-Diffraction

NumPy-based Split-Step Fourier simulator for **electron diffraction** in real SI units.  
Supports round/triangle apertures, vortex beams (OAM), multiple screens, unit tests, and Jupyter notebooks with optional animations.  
Designed for **teaching** and quick **research prototyping**.

---

## Features
- Real SI units (eV → de Broglie wavelength, k₀, Δt heuristics).
- Minimal architecture (Grid, State, Propagator, Apertures, Detectors, Runner).
- Reusable: combine multiple apertures (triangle → far zone → triangle, etc.).
- Vortex beams with orbital angular momentum (ℓ).
- Unit tests (pytest) for conservation, wavelength mapping, absorbers.
- Jupyter notebooks with Matplotlib animations.

---

## Installation
```bash
git clone https://github.com/<you>/TDSE-Diffraction.git
cd TDSE-Diffraction
pip install -e .
