"""Small geometry helpers for the observation and the rendering.

The competition generates every scenario in a flat north/east NM frame around its centre
(core/scenario._nm_to_latlon: 60 NM per degree of latitude, cos(lat0) on longitude).
to_local_nm is the exact inverse, so polygons keep their generated shape in that frame.
"""

import numpy as np


def degto180(angle):
    return (angle + 180.0) % 360.0 - 180.0


def to_local_nm(lat, lon, center):
    """(lat, lon) degrees to the scenario's flat frame: columns (north, east) in NM."""
    lat0, lon0 = center
    north = (np.asarray(lat) - lat0) * 60.0
    east = (np.asarray(lon) - lon0) * 60.0 * np.cos(np.radians(lat0))
    return np.stack([north, east], axis=-1)

