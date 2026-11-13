"""The action: a heading relative to the bearing to the goal, and an absolute speed.

Action [a0, a1] in [-1, 1]:

    heading  the bearing to the goal + a0 * 135 deg. So a0 = 0 flies straight at the goal,
             like navigation-mappo-rl's goal-aligned action. 135 deg reaches around every
             restricted area: 56% of start-goal lines cross one, and 9% of episodes need
             more than 90 deg. The turn commanded is at most 170 deg, so the aircraft never
             swings round through its tail, the short way BlueSky turns for anything past 180.
    speed    CAS 150 + 10 a1 m/s for a1 >= 0, 150 + 50 a1 m/s below: 100-160 m/s, inside the
             A320's 67-163 m/s envelope. So a1 = 0 keeps the start speed.

Action [0, 0] is therefore the competition's own fly-straight baseline: an untrained policy
already reaches its goal, and training learns when and how far to deviate for safety. The
rules allow any mapping ("map your policy's output to BlueSky heading/speed commands however
you like"). The commands are HDG and SPD, as in the shipped mapping.
"""

import numpy as np

from core.tools import kwikqdrdist

from Shared.geo import degto180

MPS2KT = 1.94384


def commands(lat, lon, hdg, goal, action, cfg):
    """The heading (deg) and CAS (m/s) to command one aircraft."""
    a0, a1 = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    qdr, _ = kwikqdrdist(lat, lon, goal[0], goal[1])
    offset_now = degto180(hdg - qdr)
    turn = np.clip(cfg['max_offset_deg'] * a0 - offset_now, -cfg['max_turn_deg'], cfg['max_turn_deg'])
    hdg_cmd = (qdr + offset_now + turn) % 360.0
    cas_cmd = cfg['cas_nominal'] + (cfg['cas_up'] if a1 >= 0 else cfg['cas_down']) * a1
    return hdg_cmd, cas_cmd


def stack_commands(bs, ac_id, goal, action, cfg):
    """Issue the commands as BlueSky HDG and SPD, the speed in knots as SPD expects."""
    i = bs.traf.id2idx(ac_id)
    hdg_cmd, cas_cmd = commands(bs.traf.lat[i], bs.traf.lon[i], bs.traf.hdg[i], goal, action, cfg)
    bs.stack.stack(f'HDG {ac_id} {hdg_cmd}')
    bs.stack.stack(f'SPD {ac_id} {cas_cmd * MPS2KT}')
