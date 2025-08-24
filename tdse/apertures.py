from __future__ import annotations
import numpy as np

# --- Plane masks (A in [0,1]) ---

def _smooth_edge(d: np.ndarray, width: float) -> np.ndarray:
    if width <= 0:
        return (d <= 0).astype(float)
    # smooth step from 1 (inside) to 0 (outside) over |d|<=width
    s = 0.5*(1 - np.tanh(d/width))
    s = np.clip(s, 0.0, 1.0)
    return s

def circle_mask(x: np.ndarray, y: np.ndarray, center: tuple[float,float], radius: float,
                smooth: float = 0.0) -> np.ndarray:
    cx, cy = center
    X, Y = np.meshgrid(x, y, indexing='ij')
    r = np.sqrt((X-cx)**2 + (Y-cy)**2)
    d = r - radius
    return _smooth_edge(d, smooth)

import numpy as np

def triangle_mask(x, y, *, center=(0.0, 0.0), side: float,
                  rotation_deg: float = 0.0, smooth: float = 0.0):
    """
    Equilateral triangle of side 'side' (meters), centered at 'center'.
    rotation_deg rotates the triangle CCW. Returns A(x,y) in [0,1] with shape (Nx,Ny).
    """
    cx, cy = center
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Vertices (equilateral), centroid at origin before rotation/shift
    Rv = side/np.sqrt(3.0)  # centroid -> vertex distance
    verts = np.array([
        [0.0,        +Rv],
        [-side/2.0,  -Rv/2.0],
        [ side/2.0,  -Rv/2.0],
    ], dtype=float)

    # Rotate and translate
    th = np.deg2rad(rotation_deg); c, s = np.cos(th), np.sin(th)
    Rm = np.array([[c, -s],[s, c]])
    verts = (verts @ Rm.T) + np.array([cx, cy])

    # Inside test via edge half-planes (vertices ordered CCW)
    # For each edge (vi -> vj), inside if (r - vi)·n_in <= 0 with n_in = rotate_cw(edge)
    def signed_distance_to_edge(Vi, Vj):
        e = Vj - Vi
        n_in = np.array([e[1], -e[0]])           # rotate CW to point inward
        elen = np.hypot(e[0], e[1])
        return ((X - Vi[0]) * n_in[0] + (Y - Vi[1]) * n_in[1]) / elen

    d1 = signed_distance_to_edge(verts[0], verts[1])
    d2 = signed_distance_to_edge(verts[1], verts[2])
    d3 = signed_distance_to_edge(verts[2], verts[0])

    if smooth <= 0:
        A = ((d1 <= 0) & (d2 <= 0) & (d3 <= 0)).astype(float)
    else:
        # Soft edge: smooth each half-plane with tanh and take min
        s1 = 0.5*(1.0 - np.tanh(d1 / smooth))
        s2 = 0.5*(1.0 - np.tanh(d2 / smooth))
        s3 = 0.5*(1.0 - np.tanh(d3 / smooth))
        A = np.minimum(np.minimum(s1, s2), s3)

    return np.clip(A, 0.0, 1.0)

# --- Apply a plane mask at z=z0 ---

def apply_plane_mask(psi: np.ndarray, grid: "Grid3D", z0: float, Axy: np.ndarray) -> None:
    kz = grid.index_of_z(z0)
    plane = psi[:, :, kz]
    if Axy.shape != plane.shape:
        raise ValueError("Mask shape must match (Nx,Ny) of the plane")
    plane *= Axy
