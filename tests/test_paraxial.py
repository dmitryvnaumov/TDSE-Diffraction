def test_paraxial_power_conservation():
    from tdse.grid2d import Grid2D
    from tdse.paraxial import ParaxialPropagator
    from tdse.units import k0_from_energy
    import numpy as np
    g = Grid2D(256,256,8e-6/256,8e-6/256)
    u = np.exp(-((g.x[:,None]**2+g.y[None,:]**2)/(1.0e-6)**2)).astype(np.complex128)
    p0 = (np.abs(u)**2).sum()*g.dx*g.dy
    prop = ParaxialPropagator(g, k0_from_energy(1e5), dz=200e-6)
    for _ in range(15): prop.step(u)
    p1 = (np.abs(u)**2).sum()*g.dx*g.dy
    assert abs(p1-p0)/p0 < 1e-12

def test_triangle_mask_area_fraction_reasonable():
    from tdse.grid2d import Grid2D
    from tdse.apertures import triangle_mask
    g = Grid2D(1024,1024,8e-6/1024,8e-6/1024)
    A = triangle_mask(g.x,g.y,center=(0,0),side=0.5e-6,rotation_deg=0.0,smooth=1e-9)
    assert A.max()>0.9
    assert 8e-4 < A.mean() < 3.5e-3   # ~0.0017 expected
