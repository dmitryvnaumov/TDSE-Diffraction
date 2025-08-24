from __future__ import annotations
import numpy as np


def make_edge_mask(grid: "Grid3D", width_cells: int = 12, power: int = 6) -> np.ndarray:
    def mask_1d(n):
        m = np.ones(n)
        ramp = np.cos(0.5*np.pi*np.linspace(0,1,width_cells))**power
        m[:width_cells] *= ramp[::-1]
        m[-width_cells:] *= ramp
        return m
    Mx = mask_1d(grid.Nx)
    My = mask_1d(grid.Ny)
    Mz = mask_1d(grid.Nz)
    return np.multiply.outer(np.multiply.outer(Mx, My), Mz).astype(np.float64)


def run(state: "State", propagator: "SplitStep", *,
        steps: int,
        V_half: np.ndarray | None = None,
        boundary_mask: np.ndarray | None = None,
        events: list[tuple[int, callable]] | None = None,
        on_measure: callable | None = None) -> None:
    """Minimal stepper.
    - events: list of (step_index, func) where func(state) applies plane masks, etc.
    - on_measure(step, state) called occasionally to record/animate.
    """
    psi = state.psi
    events = sorted(events or [], key=lambda t: t[0])
    ei = 0
    for s in range(steps):
        # apply any events scheduled BEFORE this step's kinetic
        while ei < len(events) and events[ei][0] == s:
            events[ei][1](state)
            ei += 1
        propagator.step(psi, V_half=V_half, boundary_mask=boundary_mask)
        if on_measure is not None:
            on_measure(s, state)
