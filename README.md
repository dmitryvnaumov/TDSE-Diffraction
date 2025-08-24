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
git clone https://github.com/dmitryvnaumov/TDSE-Diffraction.git
cd TDSE-Diffraction
# from the repo root
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -e .
pip install -r requirements.txt
pytest -q                   # all tests should pass

## Package Structure
TDSE-Diffraction/
├─ README.md                  # Overview, install, usage
├─ LICENSE                    # MIT or BSD
├─ pyproject.toml             # Packaging metadata (or setup.cfg)
├─ requirements.txt           # numpy, matplotlib, pytest, notebook
│
├─ tdse/                      # Core package
│  ├─ __init__.py
│  ├─ grid.py                 # Grid3D: coordinates, FFT axes
│  ├─ state.py                # State: psi container, normalize, slices
│  ├─ propagator.py           # Split-step operator, time stepping
│  ├─ apertures.py            # Circle, triangle, polygon masks
│  ├─ detectors.py            # Plane intensity, flux, radial average
│  ├─ runner.py               # Orchestration loop, events, hooks
│  ├─ units.py                # Physical constants + E↔λ mapping
│  └─ io_utils.py             # Save/load npz, png
│
├─ examples/                  # Jupyter notebooks
│  ├─ 01_round_aperture.ipynb
│  ├─ 02_triangle_cascade.ipynb
│  └─ 03_vortex_aperture.ipynb
│
├─ tests/                     # Unit tests (pytest)
│  ├─ test_free_packet.py
│  ├─ test_norm.py
│  ├─ test_absorber.py
│  └─ test_units.py
│
└─ .gitignore                 # Ignore venvs, __pycache__, data dumps

