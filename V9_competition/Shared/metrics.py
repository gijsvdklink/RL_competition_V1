"""The competition's scoring, copied from CompetitionEnv / CompetitionZooEnv._update_metrics.

The logic is copied line for line, so a surrogate episode reports the same nine metrics as a
BlueSky episode would:
- rising-edge events plus occupancy time;
- sampled once per simulated second;
- intrusions counted in both aircraft's metrics.

Distances use the deterministic core.tools.kwikqdrdist that the competition scores with.
Areas use matplotlib's Path on (lat, lon) vertices, exactly as BlueSky's POLY areas do.
The BlueSky adapters never use this class: there, the competition's own scoring layer runs.
"""

import numpy as np
from matplotlib.path import Path

from core.tools import kwikqdrdist

NM2KM = 1.852

METRIC_KEYS = ('total_reward', 'flight_time', 'waypoint_reached',
               'intrusion_events', 'intrusion_time',
               'sector_exit_events', 'time_outside_sector',
               'restricted_area_events', 'time_in_restricted_area')


class Scoring:
    """Metrics for the aircraft of one scenario, keyed by aircraft id."""

    def __init__(self, scenario, intrusion_distance_nm, distance_margin_km):
        self.sector = Path(np.array(scenario.sector))
        self.obstacles = [Path(np.array(obs.vertices)) for obs in scenario.obstacles]
        self.intrusion_distance = intrusion_distance_nm
        self.distance_margin = distance_margin_km
        self.metrics = {}
        self.waypoint_reached = {}
        self.outside_sector = {}
        self._in_intrusion_with = {}
        self._obstacles_occupied = {}

    def add(self, ac_id):
        self.metrics[ac_id] = {key: 0 if key.endswith(('events', 'reached')) else 0.0
                               for key in METRIC_KEYS}
        self.waypoint_reached[ac_id] = False
        self.outside_sector[ac_id] = False
        self._in_intrusion_with[ac_id] = set()
        self._obstacles_occupied[ac_id] = set()

    def update(self, ac_id, lat, lon, goal, others, dt=1.0):
        """others maps the id of every aircraft that can intrude to its (lat, lon)."""
        m = self.metrics[ac_id]
        m['flight_time'] += dt

        now_intr = set()
        if others:
            ids = list(others)
            olat, olon = np.array([others[ac] for ac in ids]).T
            _, d_nm = kwikqdrdist(lat, lon, olat, olon)
            now_intr = {ac for ac, d in zip(ids, d_nm) if d < self.intrusion_distance}
        m['intrusion_events'] += len(now_intr - self._in_intrusion_with[ac_id])
        m['intrusion_time'] += len(now_intr) * dt
        self._in_intrusion_with[ac_id] = now_intr

        now_obs = {i for i, area in enumerate(self.obstacles) if self._inside(area, lat, lon)}
        m['restricted_area_events'] += len(now_obs - self._obstacles_occupied[ac_id])
        if now_obs:
            m['time_in_restricted_area'] += dt
        self._obstacles_occupied[ac_id] = now_obs

        outside = not self._inside(self.sector, lat, lon)
        if outside and not self.outside_sector[ac_id]:
            m['sector_exit_events'] += 1
        if outside:
            m['time_outside_sector'] += dt
        self.outside_sector[ac_id] = outside

        _, d_goal_nm = kwikqdrdist(lat, lon, goal[0], goal[1])
        if d_goal_nm * NM2KM < self.distance_margin and not self.waypoint_reached[ac_id]:
            self.waypoint_reached[ac_id] = True
            m['waypoint_reached'] = 1

    @staticmethod
    def _inside(area, lat, lon):
        return bool(area.contains_points(np.array([[lat, lon]]))[0])
