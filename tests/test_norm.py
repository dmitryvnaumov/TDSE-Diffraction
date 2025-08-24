import numpy as np
from tdse.grid import Grid3D
from tdse.state import make_gaussian_beam, State
from tdse.propagator import SplitStep
from tdse.runner import run, make_edge_mask
from tdse.units import m_e, suggest_dt


def test_norm_conservation_small_steps():
    g = Grid3D(48,48,64, 2e-9,2e-9,2e-9)  # small box in meters
    st = make_gaussian_beam(g, E_eV=100.0, waist=10e-9, z0= -20e-9)
    dt = suggest_dt(g.dx, g.dy, g.dz, mass=m_e, phase_limit=0.2*np.pi)
    prop = SplitStep(g, m_e, dt)
    # No potential, light edge mask to avoid reflections
    M = make_edge_mask(g, 8, 6)
    n0 = st.normalize()
    def measure(_s, _state):
        pass
    run(st, prop, steps=10, V_half=None, boundary_mask=M, on_measure=measure)
    n1 = st.normalize()
    # Allow tiny drift due to masking; should be very close
    assert abs(n1 - n0) / n0 < 1e-3
