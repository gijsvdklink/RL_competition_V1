"""Measure the A320 dynamics that Shared/traffic.py imitates, in BlueSky itself.

The thesis surrogate was calibrated at FL350 in Mach. The competition creates its aircraft
at acalt=350, which BlueSky reads as 350 m, with a CAS of 150 m/s, so the constants are
measured again here, under exactly the competition's conditions:

    turn      HDG +90 from steady flight; the rate is compared with g0*tan(25)/TAS
    speed     SPD steps up and down from several speeds; the change in TAS per second
    envelope  SPD far below and far above the A320's limits; the CAS it settles at
    route     an aircraft on an ADDWPT route with a 60 deg turn; the distance to the
              waypoint at which it starts turning, against R*tan(dpsi/2)

Run from competition-rl/:

    python Validation/calibrate_dynamics.py
"""

import numpy as np
import bluesky as bs

from core.tools import kwikqdrdist

ALT_M     = 350.0
CAS_MS    = 150.0
LAT0      = 52.0
LON0      = 4.0
KT        = 0.514444
G0        = 9.80665
BANK_RAD  = np.radians(25.0)
M_PER_NM  = 1852.0


def _init_bluesky():
    bs.init(mode='sim', detached=True)
    bs.stack.stack('DT 1;FF')
    bs.stack.stack('RESO OFF')


def _fresh_aircraft(hdg=0.0, cas=CAS_MS):
    bs.traf.reset()
    bs.stack.stack('DT 1;FF')
    bs.traf.cre('CAL', actype='A320', aclat=LAT0, aclon=LON0, achdg=hdg, acspd=cas, acalt=ALT_M)
    _fly(5)


def _fly(seconds):
    rows = []
    for _ in range(int(seconds)):
        bs.sim.step()
        rows.append((bs.traf.hdg[0], bs.traf.tas[0], bs.traf.cas[0],
                     bs.traf.lat[0], bs.traf.lon[0]))
    return np.array(rows)


def measure_speeds():
    _fresh_aircraft()
    tas, cas = bs.traf.tas[0], bs.traf.cas[0]
    print(f'speeds      CAS {cas:.2f} m/s -> TAS {tas:.2f} m/s, ratio {tas / cas:.5f}')
    return tas


def measure_turn():
    _fresh_aircraft(hdg=0.0)
    tas = bs.traf.tas[0]
    bs.stack.stack('HDG CAL 90')
    hdg = _fly(80)[:, 0]
    rate = np.diff(np.concatenate([[0.0], hdg]))
    steady = rate[(rate > 0.1) & (hdg < 89.0)]
    expected = np.degrees(G0 * np.tan(BANK_RAD) / tas)
    seconds_to_90 = int(np.argmax(hdg >= 90.0 - 1e-6)) + 1
    print(f'turn        {steady.mean():.4f} deg/s measured, {expected:.4f} deg/s from '
          f'g0*tan(25)/TAS; 90 deg in {seconds_to_90} s')


def measure_acceleration():
    """OpenAP's axmax = (max thrust - drag) / mass. Level at 350 m the flight phase stays
    NA (clean), and drag does not depend on bank, so axmax is a function of TAS alone."""
    tas, axmax = [], []
    for cas in np.arange(68.0, 163.0, 2.0):
        _fresh_aircraft(cas=cas)
        tas.append(bs.traf.tas[0])
        axmax.append(bs.traf.perf.axmax[0])
    tas, axmax = np.array(tas), np.array(axmax)
    coeffs = np.polyfit(tas, axmax, 2)
    worst = np.max(np.abs(np.polyval(coeffs, tas) - axmax))
    print(f'accel       axmax {axmax.max():.3f} m/s^2 at TAS {tas[axmax.argmax()]:.1f}, '
          f'{axmax[-1]:.3f} at TAS {tas[-1]:.1f}; phase {int(bs.traf.perf.phase[0])}')
    print(f'            quadratic in TAS: {coeffs[0]:.6e}, {coeffs[1]:.6e}, {coeffs[2]:.6e}  '
          f'(worst residual {worst:.4f} m/s^2)')
    _check_acceleration_follows_axmax()


def _check_acceleration_follows_axmax():
    _fresh_aircraft()
    bs.stack.stack('SPD CAL 250')
    rows = []
    for _ in range(10):
        tas_before, axmax = bs.traf.tas[0], bs.traf.perf.axmax[0]
        bs.sim.step()
        rows.append(abs(bs.traf.tas[0] - tas_before) - axmax)
    print(f'            per-step |dTAS| - axmax over a deceleration: max {np.max(np.abs(rows)):.4f}')


def measure_envelope():
    _fresh_aircraft()
    bs.stack.stack('SPD CAL 50')
    vmin = _fly(400)[-1, 2]
    _fresh_aircraft()
    bs.stack.stack('SPD CAL 600')
    vmax = _fly(400)[-1, 2]
    print(f'envelope    CAS clipped to [{vmin:.2f}, {vmax:.2f}] m/s '
          f'= [{vmin / KT:.1f}, {vmax / KT:.1f}] kt; perf.vmin/vmax '
          f'{bs.traf.perf.vmin[0]:.2f} / {bs.traf.perf.vmax[0]:.2f}')


def measure_route_turn():
    """Fly north, then a waypoint 30 NM ahead whose next leg turns 60 deg right."""
    _fresh_aircraft(hdg=0.0)
    wp1 = (LAT0 + 30.0 / 60.0, LON0)
    leg_nm = 30.0
    brg = np.radians(60.0)
    wp2 = (wp1[0] + leg_nm * np.cos(brg) / 60.0,
           wp1[1] + leg_nm * np.sin(brg) / (60.0 * np.cos(np.radians(wp1[0]))))
    bs.stack.stack(f'CAL ADDWPT {wp1[0]} {wp1[1]}')
    bs.stack.stack(f'CAL ADDWPT {wp2[0]} {wp2[1]}')
    track = _fly(900)

    turning = np.where(np.abs(np.diff(track[:, 0])) > 0.05)[0]
    i = int(turning[0])
    _, d_nm = kwikqdrdist(track[i, 3], track[i, 4], wp1[0], wp1[1])
    tas = track[i, 1]
    radius = tas ** 2 / (G0 * np.tan(BANK_RAD))
    expected_nm = radius * np.tan(np.radians(30.0)) / M_PER_NM
    final_hdg = track[-1, 0]
    print(f'route       turn starts {d_nm:.3f} NM before the waypoint, '
          f'R*tan(30) = {expected_nm:.3f} NM; final heading {final_hdg:.2f} deg')


def main():
    _init_bluesky()
    print(f'A320 at {ALT_M:.0f} m, CAS {CAS_MS:.0f} m/s, perf model {bs.traf.perf.__class__.__name__}')
    measure_speeds()
    measure_turn()
    measure_acceleration()
    measure_envelope()
    measure_route_turn()


if __name__ == '__main__':
    main()
