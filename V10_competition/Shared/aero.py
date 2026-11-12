"""ISA atmosphere and CAS/TAS conversion, copied from bluesky.tools.aero.

Copied rather than imported so the surrogate runs without BlueSky. Only the troposphere is
needed: every aircraft flies at 350 m, where CAS 150 m/s is TAS 152.40 m/s.
"""

import numpy as np

KTS    = 0.514444        # m/s per knot
NM     = 1852.0          # m per nautical mile
G0     = 9.80665         # m/s^2
R      = 287.05287       # J/(kg K)
P0     = 101325.0        # Pa
RHO0   = 1.225           # kg/m^3
REARTH = 6371000.0       # m, BlueSky's mean earth radius


def vatmos(h):
    temp = 288.15 - 0.0065 * h
    rho = RHO0 * (temp / 288.15) ** 4.256848030018761
    return rho * R * temp, rho


def vcas2tas(cas, h):
    p, rho = vatmos(h)
    qdyn = P0 * ((1.0 + RHO0 * cas * cas / (7.0 * P0)) ** 3.5 - 1.0)
    return np.sqrt(7.0 * p / rho * ((1.0 + qdyn / p) ** (2.0 / 7.0) - 1.0))


def vtas2cas(tas, h):
    p, rho = vatmos(h)
    qdyn = p * ((1.0 + rho * tas * tas / (7.0 * p)) ** 3.5 - 1.0)
    return np.sqrt(7.0 * P0 / RHO0 * ((qdyn / P0 + 1.0) ** (2.0 / 7.0) - 1.0))
