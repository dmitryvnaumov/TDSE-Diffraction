from typing import Callable, List, Tuple
import numpy as np

def run2d(u_xy, prop, *, steps, events=None, on_measure=None):
    events = sorted(events or [], key=lambda t: t[0])
    ei = 0
    for s in range(steps):
        # measure at current plane (z = s*dz)
        if on_measure is not None:
            on_measure(s, u_xy)
        while ei < len(events) and events[ei][0] == s:
            events[ei][1](u_xy)
            ei += 1
        prop.step(u_xy)

