"""Top-down frames of an episode, drawn with the competition's own core.rendering.

It draws what CompetitionEnv._draw_world draws:
- the sector, the obstacles and the intruder routes;
- every aircraft with its 5 NM circle, red while in loss of separation;
- the goals.

On top of that come the lidar hits of every controlled aircraft, decoded from the newest
snapshot of its observation and coloured by channel. The goal channel is not drawn; the goal
itself is. The renderer reads a TrafficView, so
the same code draws the surrogate and BlueSky.
"""

import os

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

import numpy as np
import pygame

from core.rendering import (PygameCanvas, TopDownProjection, draw_intruder, draw_line,
                            draw_polygon, draw_waypoint, BRIGHT_GREEN, LIGHT_GRAY, SLATE_GRAY)
from core.tools import kwikqdrdist

from Shared.geo import to_local_nm
from Shared.observation import N_CHANNELS, STATE_DIM, frame_dim

NM2KM = 1.852
CHANNEL_COLOURS = ((255, 140, 0), (3, 160, 11), (220, 20, 60))     # obstacle, boundary, aircraft
OWN_COLOUR = (0, 0, 0)


class Renderer:

    def __init__(self, scenario, cfg, size=512):
        self.scenario, self.cfg = scenario, cfg
        self.canvas = PygameCanvas(size, size, mode='rgb_array')
        sector_nm = to_local_nm(*np.array(scenario.sector).T, scenario.center)
        span_nm = np.hypot(np.ptp(sector_nm[:, 0]), np.ptp(sector_nm[:, 1]))
        self.projection = TopDownProjection(max_distance=span_nm * NM2KM * 1.1,
                                           ref_lat=scenario.center[0], ref_lon=scenario.center[1],
                                           window_size=(size, size))
        fov = cfg['fov_deg']
        self.offsets = np.radians(np.linspace(-fov / 2, fov / 2, cfg['n_rays'], endpoint=fov < 360.0))

    def frame(self, traffic, controlled, goals, observations=None):
        canvas = self.canvas.begin_frame()
        project = self.projection.project
        draw_polygon(canvas, [project(*v) for v in self.scenario.sector],
                     color=BRIGHT_GREEN, filled=False, width=2)
        for obstacle in self.scenario.obstacles:
            draw_polygon(canvas, [project(*v) for v in obstacle.vertices], color=SLATE_GRAY)
        for route in self.scenario.intruder_routes:
            points = [project(*w) for w in route.waypoints]
            for a, b in zip(points[:-1], points[1:]):
                draw_line(canvas, *a, *b, color=LIGHT_GRAY)
        for ac in controlled:
            draw_waypoint(canvas, *project(*goals[ac]), margin_km=self.cfg['distance_margin_km'],
                          projection=self.projection)
        for ac, obs in (observations or {}).items():
            if ac in traffic.id:
                self._draw_lidar(canvas, traffic, traffic.id.index(ac), obs)
        self._draw_aircraft(canvas, traffic, controlled)
        return self.canvas.end_frame(canvas)

    def _draw_aircraft(self, canvas, traffic, controlled):
        project = self.projection.project
        sep = self.cfg['intrusion_distance_nm']
        _, dist = kwikqdrdist(traffic.lat[:, None], traffic.lon[:, None],
                              traffic.lat[None, :], traffic.lon[None, :])
        np.fill_diagonal(dist, np.inf)
        for i, ac in enumerate(traffic.id):
            own = ac in controlled
            draw_intruder(canvas, *project(traffic.lat[i], traffic.lon[i]), traffic.hdg[i],
                          self.projection, body_km=6 if own else 2, heading_km=12 if own else 5,
                          safety_radius_km=sep * NM2KM / 2 if own else sep * NM2KM,
                          in_intrusion=bool((dist[i] < sep).any()),
                          color_safe=OWN_COLOUR if own else (80, 80, 80))

    def _draw_lidar(self, canvas, traffic, row, observation):
        newest = observation[-frame_dim(self.cfg):]
        scan = newest[STATE_DIM:].reshape(N_CHANNELS, self.cfg['n_rays'])
        origin_nm = to_local_nm(traffic.lat[row], traffic.lon[row], self.scenario.center)
        angles = np.radians(traffic.hdg[row]) + self.offsets
        x0, y0 = self.projection.project(traffic.lat[row], traffic.lon[row])
        for channel, ray in zip(*np.nonzero(scan[:3])):
            dist = self.cfg['lidar_range_nm'] * (1.0 - scan[channel, ray])
            north, east = origin_nm + dist * np.array([np.cos(angles[ray]), np.sin(angles[ray])])
            lat, lon = self._to_latlon(north, east)
            x, y = self.projection.project(lat, lon)
            draw_line(canvas, x0, y0, x, y, color=CHANNEL_COLOURS[channel])
            pygame.draw.circle(canvas, CHANNEL_COLOURS[channel], (int(x), int(y)), 2)

    def _to_latlon(self, north, east):
        lat0, lon0 = self.scenario.center
        return lat0 + north / 60.0, lon0 + east / (60.0 * np.cos(np.radians(lat0)))
