"""The reward per decision step (10 s), purely negative, in the style of navigation-mappo-rl:

    r = -(10 new_LoS + 10 new_restricted + 10 new_exit + 5/300 + 12/300 * (1 - cos drift))

The first three terms are 1 when the matching scored event count grew during the step: a new
loss of separation, a new entry into a restricted area, a new exit from the sector, read from
the competition's own metrics. They weigh the same, so a squeeze never teaches the agent to
trade one for another. In training a new violation also ends the aircraft's episode, as a
collision does in the repo, where every collision costs -10 (terminate_on_violation).
Evaluation keeps the competition's rules: nothing terminates on a violation there. A loss
of separation that already exists when the episode starts was counted at reset, so it
neither costs nor terminates; leaving it and coming back does.

Time costs 5/300 = 0.017 per decision, -5 over a full 3000 s episode. drift is the angle
between the heading and the bearing to the goal at the end of the step, as in the
competition's base reward, and 1 - cos(drift) is 0 flying straight at the goal, 1 across it,
2 away from it: 0.04 per decision at 90 deg.

The weights balance two failures, judged by PPO's discounted view (gamma 0.98: costs count
for about 1/(1 - gamma) = 50 decisions; a flight to the goal takes about 100):

    no donuts   circling forever costs 50 (t + d) = 2.8, flying to the goal 43 t = 0.7, so
                arriving is worth 2.1: the agent goes even if that is 21% riskier. Arrival
                is 100 decisions away, so the time penalty hardly counts against circling;
                the drift, paid at once, does.
    no crashing on purpose  the worst continuation, flying away, costs 50 (t + 2d) = 4.8,
                half of the 10 a violation costs, so ending the episode never pays.

Raising the time penalty instead fails both: at 10 per episode no drift weight satisfies
both conditions. A 200 s detour of 45 deg costs 0.23, far below a violation.
"""

import numpy as np

from core.tools import kwikqdrdist

from Shared.geo import degto180

EVENTS = ('intrusion_events', 'restricted_area_events', 'sector_exit_events')


def new_violation(now, before):
    """True when the step brought a new loss of separation, restricted entry or exit."""
    return any(now[key] > before[key] for key in EVENTS)


def drift(lat, lon, hdg, goal):
    """The angle in radians between the heading and the bearing to the goal."""
    qdr, _ = kwikqdrdist(lat, lon, goal[0], goal[1])
    return abs(np.radians(degto180(hdg - qdr)))


def reward(now, before, drift_rad, cfg):
    """now and before are one aircraft's metric dicts at the end and start of the step."""
    los = now['intrusion_events'] > before['intrusion_events']
    restricted = now['restricted_area_events'] > before['restricted_area_events']
    outside = now['sector_exit_events'] > before['sector_exit_events']
    per_step = cfg['action_frequency'] / cfg['episode_time_limit_s']
    return -(cfg['w_los'] * los
             + cfg['w_restricted'] * restricted
             + cfg['w_outside'] * outside
             + per_step * (cfg['w_time_episode'] + cfg['w_drift_episode'] * (1.0 - np.cos(drift_rad))))
