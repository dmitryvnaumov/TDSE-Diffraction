#!/usr/bin/env python3
"""
Paraxial (2D FFT) propagation of an OAM (vortex) electron beam
through a triangular aperture; display intensity at several far‑zone z planes.

Usage examples:
  python examples/triangle_paraxial_cli.py
  python examples/triangle_paraxial_cli.py --E_keV 100 --ell 5 --side_um 0.5 --z_um 200,500,1000
  python examples/triangle_paraxial_cli.py --Nx 768 --FOV_um 12 --rot_deg 30 --save_prefix out/tri

The script relies on the lightweight 2D API:
  - tdse.Grid2D
  - tdse.ParaxialPropagator
  - tdse.apply_mask_2d
  - tdse.run2d
  - tdse.units.k0_from_energy
  - tdse.apertures.triangle_mask
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from tdse import Grid2D, ParaxialPropagator, apply_mask_2d, run2d
from tdse.units import k0_from_energy
from tdse.apertures import triangle_mask

def make_oam_envelope(x, y, waist_m, ell):
    """Return complex 2D OAM (LG p=0-like) envelope at z=0."""
    X, Y = np.meshgrid(x, y, indexing="ij")
    rho2 = X**2 + Y**2
    phi = np.arctan2(Y, X)
    return (np.sqrt(rho2)/waist_m)**abs(ell) * np.exp(-rho2/waist_m**2) * np.exp(1j*ell*phi)

def main():
    ap = argparse.ArgumentParser(description="Triangle aperture: paraxial FFT propagation (2D).")
    ap.add_argument("--E_keV", type=float, default=100.0, help="Electron kinetic energy (keV). Default: 100")
    ap.add_argument("--ell", type=int, default=5, help="OAM topological charge ℓ. Default: 5")
    ap.add_argument("--side_um", type=float, default=0.5, help="Triangle side length (µm). Default: 0.5")
    ap.add_argument("--rot_deg", type=float, default=0.0, help="Triangle rotation angle (degrees).")
    ap.add_argument("--FOV_um", type=float, default=8.0, help="Field of view side length (µm). Default: 8.0")
    ap.add_argument("--Nx", type=int, default=512, help="Grid points in x. Default: 512")
    ap.add_argument("--Ny", type=int, default=None, help="Grid points in y (default: Nx)")
    ap.add_argument("--waist_um", type=float, default=1.0, help="OAM beam waist (µm). Default: 1.0")
    ap.add_argument("--smooth_nm", type=float, default=10.0, help="Edge smoothing for mask (nm). Default: 10")
    ap.add_argument("--dz_um", type=float, default=200.0, help="Propagation step (µm). Default: 200")
    ap.add_argument("--z_um", type=str, default="200,600,2000",
                    help="Comma‑separated z planes (µm) to show, e.g. 200,600,2000")
    ap.add_argument("--save_prefix", type=str, default="",
                    help="If set, save PNGs with this prefix (e.g., 'out/tri').")
    ap.add_argument("--auto", action="store_true",
                help="Auto-pick FOV, dz, and z planes near the Fraunhofer scale.")
    ap.add_argument("--auto_view", action="store_true", default=True,
                help="Auto-zoom each plane to include most power (default on).")
    ap.add_argument("--view_frac", type=float, default=0.95,
                    help="Fraction of total power to include in the auto view (0–1). Default: 0.95")
    ap.add_argument("--pad", type=float, default=1.2,
                    help="Padding factor on the computed view radius. Default: 1.2")
    ap.add_argument("--same_axes", action="store_true",
                    help="Use the same axis limits for all panels (based on the largest auto view).")


    args = ap.parse_args()

    # --- Units & derived quantities (SI) ---
    E_eV = args.E_keV * 1e3
    k0 = k0_from_energy(E_eV)  # rad/m
    side = args.side_um * 1e-6
    FOV = args.FOV_um * 1e-6
    waist = args.waist_um * 1e-6
    smooth = args.smooth_nm * 1e-9
    dz = args.dz_um * 1e-6
    z_list = [float(z.strip()) * 1e-6 for z in args.z_um.split(",") if z.strip()]

    Nx = args.Nx
    Ny = args.Ny if args.Ny is not None else Nx

    lam = 2*np.pi/k0
    theta = lam/side
    px = (args.FOV_um*1e-6)/ (args.Nx)
    print(f"λ={lam*1e12:.3f} pm, θ≈{theta*1e6:.2f} µrad, pixel={px*1e9:.2f} nm")
    if side/px < 10:
        print(f"[warn] Triangle spans only {side/px:.1f} px; increase Nx or reduce FOV.")
    for z in z_list:
        r = z*theta
        if r/px < 3:
            print(f"[warn] z={z*1e6:.0f} µm → central lobe {r/px:.1f} px (<3): pattern may look flat.")


    # --- Grid & propagator ---
    dx = FOV / Nx
    dy = FOV / Ny
    g = Grid2D(Nx, Ny, dx, dy)
    prop = ParaxialPropagator(g, k0=k0, dz=dz)

    # --- Initial field u(x,y,0) = OAM envelope × triangle mask (thin screen at z=0) ---
    A = triangle_mask(g.x, g.y, center=(0.0, 0.0), side=side, rotation_deg=args.rot_deg, smooth=smooth)
    print("mask stats: min", A.min(), "max", A.max(), "mean", A.mean())
    plt.imshow(A.T, origin="lower"); plt.title("triangle mask"); plt.colorbar(); plt.show()

    u = make_oam_envelope(g.x, g.y, waist_m=waist, ell=args.ell).astype(np.complex128)
    apply_mask_2d(u, A)
    plt.imshow((np.abs(u)**2).T, origin="lower"); plt.title("|u|^2 at z=0"); plt.colorbar(); plt.show()

    # Map requested z to step indices for pre-step measurement
    target_steps = sorted({ int(round(z/dz)) for z in z_list })
    max_steps = (max(target_steps) + 1) if target_steps else 1

    planes = {}

    def on_measure(step, u_xy):
        if step not in target_steps or step in planes:
            return

        I = np.abs(u_xy)**2
        I /= I.max() + 1e-30
        z_here = step * dz  # pre-step: field corresponds to z = step*dz

        crop = None
        if args.auto_view:
            # radius enclosing 'view_frac' of total power
            X, Y = np.meshgrid(g.x, g.y, indexing="ij")
            R = np.hypot(X, Y)
            r_flat = R.ravel()
            w_flat = (I / (I.sum() + 1e-30)).ravel()
            order = np.argsort(r_flat)
            csum = np.cumsum(w_flat[order])
            idx = np.searchsorted(csum, args.view_frac, side="left")
            r_q = r_flat[order][min(idx, len(order)-1)]
            r_view = args.pad * max(r_q, 3*max(g.dx, g.dy))  # avoid tiny window

            # indices for crop; clamp if r_view exceeds FOV
            ix = np.where((g.x >= -r_view) & (g.x <= r_view))[0]
            iy = np.where((g.y >= -r_view) & (g.y <= r_view))[0]
            if len(ix) and len(iy):
                crop = (ix[0], ix[-1], iy[0], iy[-1])
            else:
                crop = None  # fall back to full FOV

        # store exactly once, including crop if any
        planes[step] = (z_here, I.copy(), crop)


    run2d(u, prop, steps=max_steps, on_measure=on_measure)

    # --- Report & plots ---
    lam = 2*np.pi / k0
    print(f"Energy: {args.E_keV:.2f} keV  |  λ = {lam*1e12:.3f} pm  |  k0 = {k0:.3e} rad/m")
    print(f"Grid: {Nx}×{Ny}, FOV = {args.FOV_um:.2f} µm, dx = {dx*1e9:.2f} nm")
    print(f"Triangle side = {args.side_um:.3f} µm, rotation = {args.rot_deg:.1f}°")
    print(f"dz = {args.dz_um:.1f} µm; planes @ " +
          ", ".join(f"{(s*dz)*1e6:.0f} µm" for s in target_steps))

    # determine shared axes if requested
    shared_extent = None
    if args.same_axes:
        xmin, xmax = g.x[0], g.x[-1]
        ymin, ymax = g.y[0], g.y[-1]
        any_crop = False
        for _, (_, _, crop) in sorted(planes.items()):
            if crop is not None:
                any_crop = True
                i0, i1, j0, j1 = crop
                xmin = min(xmin, g.x[i0]); xmax = max(xmax, g.x[i1])
                ymin = min(ymin, g.y[j0]); ymax = max(ymax, g.y[j1])
        # if no crops, shared_extent stays as the full FOV
        shared_extent = [xmin*1e6, xmax*1e6, ymin*1e6, ymax*1e6]

    ncols = len(planes) if planes else 1
    fig, axes = plt.subplots(1, ncols, figsize=(4*ncols, 4), constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    for ax, step in zip(axes, sorted(planes.keys())):
        z_here, I, crop = planes[step]
        if args.auto_view and (crop is not None) and (not args.same_axes):
            i0, i1, j0, j1 = crop
            Iplot = I[i0:i1+1, j0:j1+1]
            extent = [g.x[i0]*1e6, g.x[i1]*1e6, g.y[j0]*1e6, g.y[j1]*1e6]
        else:
            Iplot = I
            extent = (shared_extent if args.same_axes
                    else [g.x[0]*1e6, g.x[-1]*1e6, g.y[0]*1e6, g.y[-1]*1e6])

        im = ax.imshow(Iplot.T, origin="lower", extent=extent,
                    aspect="equal", vmin=0, vmax=1)

        # Force axis limits to the extent (sometimes helpful with colorbar/layout)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        ax.set_title(f"z ≈ {z_here*1e6:.0f} µm")
        ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
        fig.colorbar(im, ax=ax, shrink=0.85, label="norm. intensity")

    plt.suptitle(f"Paraxial triangle  |  ℓ={args.ell}  |  E={args.E_keV:.0f} keV  |  side={args.side_um:.2f} µm")
    plt.show()

    # --- Optional saves ---
    if args.save_prefix:
        os.makedirs(os.path.dirname(args.save_prefix), exist_ok=True) if os.path.dirname(args.save_prefix) else None
        for step in sorted(planes.keys()):
            z_here, I, crop = planes[step]

            if args.auto_view and (crop is not None) and (not args.same_axes):
                i0, i1, j0, j1 = crop
                Iplot = I[i0:i1+1, j0:j1+1]
                extent_save = [g.x[i0]*1e6, g.x[i1]*1e6, g.y[j0]*1e6, g.y[j1]*1e6]
            else:
                Iplot = I
                extent_save = (shared_extent if args.same_axes
                            else [g.x[0]*1e6, g.x[-1]*1e6, g.y[0]*1e6, g.y[-1]*1e6])

            out = f"{args.save_prefix}_z{int(round(z_here*1e6))}um.png"
            plt.figure(figsize=(4.5,4))
            plt.imshow(Iplot.T, origin="lower", extent=extent_save, aspect="equal", vmin=0, vmax=1)
            plt.xlim(extent_save[0], extent_save[1])
            plt.ylim(extent_save[2], extent_save[3])
            plt.xlabel("x (µm)"); plt.ylabel("y (µm)")
            plt.title(f"z ≈ {z_here*1e6:.0f} µm")
            plt.colorbar(label="norm. intensity", shrink=0.85)
            plt.tight_layout()
            plt.savefig(out, dpi=200)
            plt.close()
            print(f"Saved {out}")

    if args.auto:
        lam = 2*np.pi/k0
        zF = side*side/lam
        z_targets = [0.3*zF, 1.0*zF, 1.3*zF]
        r_max = max(z_targets) * lam / side
        FOV = 4.0 * r_max
        dz = max(z_targets) / 16.0
        args.FOV_um = FOV*1e6
        args.dz_um  = dz*1e6
        args.z_um   = ",".join(f"{z*1e6:.0f}" for z in z_targets)
        print(f"[auto] FOV={args.FOV_um:.1f} µm, dz={args.dz_um:.1f} µm, planes={args.z_um}")
        

if __name__ == "__main__":
    main()
