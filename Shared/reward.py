"""The reward per decision step, purely negative:

    r = -(100 LoS + 10 in_restricted + 10 outside_sector + 1 * (1 - cos drift) + 0.1)

The first three terms are binary: each is 1 when the matching scored metric's occupancy time
grew during the step, that is, when at least one second of the step was spent in loss of
separation, in a restricted area, or outside the sector.

drift is the angle between the heading and the bearing to the goal at the end of the step,
as in the competition's own base reward. 1 - cos(drift) is:
- 0 when flying straight at the goal, so heading for it costs nothing;
- 0.29 at 45 deg off, 1 when flying perpendicular to it, 2 when flying away.

Circling therefore costs about 1 per step, roughly -100 over the discount horizon, so
reaching the goal is worth the effort. A detour around traffic or a restricted area costs
only a few points, against 100 for every step in loss of separation.

The last term is a time penalty of 0.1 per decision step. Pointing at the goal alone is
free, so without it a slow flight would cost nothing; with it, the sooner the goal is
reached the better. There is no goal bonus: reaching the goal ends the episode, and with
it the penalties.
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
    los = now['intrusion_time'] > before['intrusion_time']
    restricted = now['time_in_restricted_area'] > before['time_in_restricted_area']
    outside = now['time_outside_sector'] > before['time_outside_sector']
    return -(cfg['w_los'] * los
             + cfg['w_restricted'] * restricted
             + cfg['w_outside'] * outside
             + cfg['w_drift'] * (1.0 - np.cos(drift_rad))
             + cfg['w_time'])
