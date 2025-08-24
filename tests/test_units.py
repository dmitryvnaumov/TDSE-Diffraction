import numpy as np
from tdse.units import electron_wavelength, k0_from_energy, q_e, m_e, c, h

def test_electron_wavelength_100eV_reasonable():
    lam = electron_wavelength(100.0)  # meters
    # Known: 100 eV electrons ~ 0.123 nm (nonrelativistic 0.1239 nm), relativistic ~0.122 nm
    assert 0.08e-9 < lam < 0.20e-9
    k0 = k0_from_energy(100.0)
    assert np.isclose(k0, 2*np.pi/lam)