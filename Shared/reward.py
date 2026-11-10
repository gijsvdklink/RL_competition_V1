"""The reward per decision step, purely negative and summed over its simulated seconds:

    r = -(100 LoS + 10 in_restricted + 10 outside_sector + 1 * (1 - cos drift) + 0.1) per second

The first three terms count seconds, read from how much the matching scored metric grew
during the step:
- LoS is intrusion_time, so 5 s within 5 NM of one aircraft costs 500, and of two
  aircraft at once 1000, exactly as the competition counts it;
- in_restricted is time_in_restricted_area, so 5 s inside a restricted area costs 50;
- outside_sector is time_outside_sector, so 5 s outside the sector costs 50.

The reward therefore reads only the committed scoring state, never the observation, and is
the same in the surrogate and in BlueSky.

drift is the angle between the heading and the bearing to the goal at the end of the step, as
in the competition's own base reward. 1 - cos(drift) is:
- 0 when flying straight at the goal;
- 0.29 at 45 deg off, 1 when flying perpendicular to it, 2 when flying away.

Together with the time penalty of 0.1 per second, a 5 s step costs 0.5 when flying straight
at the goal and about 5.5 when circling. The sooner the goal is reached the better, and a
detour around traffic still costs far less than a second in loss of separation. There is no
goal bonus: reaching the goal ends the episode, and only the seconds flown in that last step
count.
"""

import numpy as np

from core.tools import kwikqdrdist

from Shared.geo import degto180


def drift(lat, lon, hdg, goal):
    """The angle in radians between the heading and the bearing to the goal."""
    qdr, _ = kwikqdrdist(lat, lon, goal[0], goal[1])
    return abs(np.radians(degto180(hdg - qdr)))


def reward(now, before, drift_rad, cfg):
    """now and before are one aircraft's metric dicts at the end and start of the step."""
    def grew(key):
        return now[key] - before[key]

    return -(cfg['w_los'] * grew('intrusion_time')
             + cfg['w_restricted'] * grew('time_in_restricted_area')
             + cfg['w_outside'] * grew('time_outside_sector')
             + (cfg['w_drift'] * (1.0 - np.cos(drift_rad)) + cfg['w_time']) * grew('flight_time'))
