"""A kinematic traffic model in place of BlueSky, for faster training.

A port of the thesis surrogate (MAIN_FINAL_NO_BS/Environment/traffic.py), re-measured for the
competition's aircraft: an A320 at 350 m with a CAS of 150 m/s (TAS 152.40 m/s).
Validation/calibrate_dynamics.py repeats every measurement:

    turns     BlueSky holds a fixed 25 deg bank, so the rate is g0*tan(25)/TAS: 1.7192 deg/s
              at the start speed, 53 s for 90 deg. It is recomputed from the current TAS.
    speed     TAS moves toward the commanded CAS at OpenAP's axmax, which at this altitude
              depends on TAS alone: 1.47 m/s^2 at 83 m/s falling to 0.88 at 165 m/s, fitted by
              a quadratic (worst residual 0.04 m/s^2). The commanded CAS is clipped to the
              envelope [67, 163] m/s first, as perf.limits does.
    position  lat/lon integrated exactly as BlueSky's update_pos, so every distance is scored in
              the same frame as the competition's kwikqdrdist.
    routes    intruders steer straight at their active waypoint (BlueSky LNAV) and switch at
              the fly-by distance R*tan(dpsi/2), or once they pass it; the turn started 1.547 NM
              before a 60 deg waypoint against 1.583 NM predicted, one 0.08 NM step apart.

Each step runs in BlueSky's order: route logic, speed, then a turn at the new speed, then
position. There is no global state, so each environment owns its Traffic, where BlueSky is a
process-wide singleton. The attribute names mirror the BlueSky fields the environments read.
"""

import numpy as np

from core.tools import kwikqdrdist

from Shared.aero import G0, NM, REARTH, vcas2tas, vtas2cas

BANK_RAD      = np.radians(25.0)
VMIN_CAS      = 67.0
VMAX_CAS      = 163.0
AXMAX_POLY    = (-7.117481e-05, 1.017673e-02, 1.106390e+00)
AXMAX_FLOOR   = 0.5
CLOSE_TO_WP_S = 4.0

_ARRAYS = ('lat', 'lon', 'hdg', 'tas', 'cas', 'cmd_hdg', 'cmd_cas',
           'swlnav', 'wp_lat', 'wp_lon', 'legdir', 'turndist')


def degto180(angle):
    return (angle + 180.0) % 360.0 - 180.0


class Traffic:
    """The aircraft of one environment, integrated every simulated second.

    Positions are lat/lon degrees, speeds m/s, headings degrees clockwise from north.
    cmd_hdg and cmd_cas are the autopilot targets, BlueSky's ap.trk and selspd.
    """

    __slots__ = ('alt', 'id', '_route', '_wp_index') + _ARRAYS

    def __init__(self, altitude_m):
        self.alt = altitude_m
        self.reset()

    def reset(self):
        self.id = []
        self._route = []
        self._wp_index = []
        for name in _ARRAYS:
            setattr(self, name, np.zeros(0, dtype=bool if name == 'swlnav' else float))

    @property
    def ntraf(self):
        return len(self.id)

    def id2idx(self, acid):
        return self.id.index(acid) if acid in self.id else -1

    def create(self, acid, lat, lon, hdg, cas):
        tas = float(vcas2tas(cas, self.alt))
        values = dict(lat=lat, lon=lon, hdg=hdg % 360.0, tas=tas, cas=cas,
                      cmd_hdg=hdg % 360.0, cmd_cas=cas, swlnav=False,
                      wp_lat=0.0, wp_lon=0.0, legdir=0.0, turndist=0.0)
        for name in _ARRAYS:
            setattr(self, name, np.append(getattr(self, name), values[name]))
        self.id.append(acid)
        self._route.append(np.zeros((0, 2)))
        self._wp_index.append(0)

    def delete(self, acid):
        i = self.id2idx(acid)
        if i < 0:
            return
        for name in _ARRAYS:
            setattr(self, name, np.delete(getattr(self, name), i))
        del self.id[i], self._route[i], self._wp_index[i]

    def set_heading(self, acid, hdg):
        """HDG command: a new heading target, and LNAV off."""
        i = self.id2idx(acid)
        self.cmd_hdg[i] = hdg % 360.0
        self.swlnav[i] = False

    def set_speed(self, acid, cas):
        """SPD command: a new CAS target in m/s."""
        self.cmd_cas[self.id2idx(acid)] = cas

    def add_route(self, acid, waypoints):
        """ADDWPT for every (lat, lon); LNAV on, direct to the first."""
        i = self.id2idx(acid)
        self._route[i] = np.asarray(waypoints, dtype=float)
        self._wp_index[i] = 0
        self.swlnav[i] = True
        self._activate(i)

    def step(self, dt=1.0):
        if not self.id:
            return
        self._lnav()
        self._update_airspeed(dt)
        self._update_heading(dt)
        self._update_pos(dt)

    # ---------------------------------------------------------------------------- routes
    def _lnav(self):
        on = np.flatnonzero(self.swlnav)
        if not on.size:
            return
        for i in on[self._reached(on)]:
            self._wp_index[i] += 1
            self._activate(i)
        on = np.flatnonzero(self.swlnav)
        qdr, _ = kwikqdrdist(self.lat[on], self.lon[on], self.wp_lat[on], self.wp_lon[on])
        self.cmd_hdg[on] = qdr

    def _reached(self, on):
        """BlueSky's ActiveWaypoint.reached: inside the turn distance, or passed the waypoint."""
        qdr, dist_nm = kwikqdrdist(self.lat[on], self.lon[on], self.wp_lat[on], self.wp_lon[on])
        dist = dist_nm * NM
        flying_away = np.abs(degto180(self.hdg[on] - qdr)) > 90.0
        too_close = (dist / np.maximum(self.tas[on], 1e-4) < CLOSE_TO_WP_S) & flying_away
        passed = np.abs(degto180(qdr - self.legdir[on])) > 90.0
        return too_close | passed | (dist < self.turndist[on])

    def _activate(self, i):
        """Make waypoint _wp_index[i] active; past the last one LNAV switches off."""
        route, k = self._route[i], self._wp_index[i]
        if k >= len(route):
            self.swlnav[i] = False
            return
        self.wp_lat[i], self.wp_lon[i] = route[k]
        self.legdir[i], _ = kwikqdrdist(self.lat[i], self.lon[i], route[k, 0], route[k, 1])
        next_qdr = self.legdir[i]
        if k + 1 < len(route):
            next_qdr, _ = kwikqdrdist(route[k, 0], route[k, 1], route[k + 1, 0], route[k + 1, 1])
        turnrad = self.tas[i] ** 2 / (G0 * np.tan(BANK_RAD))
        self.turndist[i] = abs(turnrad * np.tan(np.radians(0.5 * abs(degto180(self.legdir[i] - next_qdr)))))

    # ----------------------------------------------------------------------------- motion
    def _update_airspeed(self, dt):
        target = vcas2tas(np.clip(self.cmd_cas, VMIN_CAS, VMAX_CAS), self.alt)
        axmax = np.maximum(np.polyval(AXMAX_POLY, self.tas), AXMAX_FLOOR)
        delta = target - self.tas
        need_ax = np.abs(delta) > dt * axmax
        self.tas = np.where(need_ax, self.tas + np.sign(delta) * axmax * dt, target)
        self.cas = vtas2cas(self.tas, self.alt)

    def _update_heading(self, dt):
        turnrate = np.degrees(G0 * np.tan(BANK_RAD) / np.maximum(self.tas, 1e-4))
        delhdg = degto180(self.cmd_hdg - self.hdg)
        turning = np.abs(delhdg) > dt * turnrate
        self.hdg = np.where(turning, self.hdg + dt * turnrate * np.sign(delhdg), self.cmd_hdg) % 360.0

    def _update_pos(self, dt):
        hdg = np.radians(self.hdg)
        self.lat = self.lat + np.degrees(dt * self.tas * np.cos(hdg) / REARTH)
        coslat = np.cos(np.radians(self.lat))
        self.lon = self.lon + np.degrees(dt * self.tas * np.sin(hdg) / coslat / REARTH)
