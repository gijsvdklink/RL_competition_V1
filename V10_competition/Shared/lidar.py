"""A three-channel range scan around each aircraft (V2).

Every aircraft casts n_rays rays, spread over fov_deg and centred on its own heading, in
the scenario's flat north/east NM frame. Three channels:

    0  restricted airspace OR sector boundary   the obstacle and sector polygon edges
    1  other aircraft     discs of aircraft_radius_nm (2.5 NM, half the separation minimum)
    2  destination        the goal's capture disc (radius distance_margin, 5 km)

Channels 0 and 1 compete: a ray reports only the nearest of the two, the other is 0.
Channel 2 is cast on its own, so the destination never hides an obstacle or an aircraft.
The value is (range - hit distance) / range: 0 for nothing within range, rising to 1 at
contact. The ray-segment and ray-circle tests are the vectorised ones from
navigation-mappo-rl/nav/ray_intersection.py.
"""

import numpy as np


class Lidar:

    def __init__(self, n_rays, fov_deg, range_nm, aircraft_radius_nm, goal_radius_nm):
        endpoint = fov_deg < 360.0
        self.offsets = np.radians(np.linspace(-fov_deg / 2, fov_deg / 2, n_rays, endpoint=endpoint))
        self.n_rays = n_rays
        self.range = range_nm
        self.radius = aircraft_radius_nm
        self.goal_radius = goal_radius_nm
        self.edges = np.zeros((0, 4))
        self.n_obstacle_edges = 0

    def set_scene(self, sector_nm, obstacles_nm):
        """Polygons as (n, 2) arrays of (north, east) NM vertices, unclosed. The obstacle
        edges come first in one array with the sector's, so one call casts against both."""
        obstacle_edges = [_edges(p) for p in obstacles_nm]
        self.n_obstacle_edges = sum(len(e) for e in obstacle_edges)
        self.edges = np.concatenate(obstacle_edges + [_edges(sector_nm)])

    def scan(self, pos_nm, hdg_deg, aircraft_nm, own_index, goal_nm):
        """Scan from every row of pos_nm; returns (n_scanners, 3, n_rays) in [0, 1].

        aircraft_nm holds every aircraft that shows on the scan, own_index the row of each
        scanner in it, so a scanner does not see itself. goal_nm is each scanner's own goal.
        """
        n = len(pos_nm)
        angles = (np.radians(hdg_deg)[:, None] + self.offsets[None, :]).ravel()
        origins = np.repeat(pos_nm, self.n_rays, axis=0)
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)

        own = np.repeat(own_index, self.n_rays)
        to_aircraft = _ray_circle(origins, directions, self.range, aircraft_nm, self.radius)
        to_aircraft[np.arange(len(own)), own] = np.inf

        to_edges = _ray_segment(origins, directions, self.range, self.edges)
        hits = np.stack([_nearest(to_edges), _nearest(to_aircraft)])
        channel = hits.argmin(axis=0)
        dist = hits.min(axis=0)

        scan = np.zeros((3, n * self.n_rays))
        hit = np.isfinite(dist)
        scan[channel[hit], np.flatnonzero(hit)] = (self.range - dist[hit]) / self.range

        # each scanner's rays against its own goal only
        goals = np.repeat(np.asarray(goal_nm, dtype=float), self.n_rays, axis=0)
        to_goal = _ray_circle(origins, directions, self.range, goals, self.goal_radius)
        to_goal = to_goal[np.arange(len(goals)), np.arange(len(goals))]
        seen = np.isfinite(to_goal)
        scan[2, seen] = (self.range - to_goal[seen]) / self.range
        return scan.reshape(3, n, self.n_rays).transpose(1, 0, 2)


def _edges(vertices):
    v = np.asarray(vertices, dtype=float)
    return np.concatenate([v, np.roll(v, -1, axis=0)], axis=1)


def _nearest(distances):
    if distances.shape[1] == 0:
        return np.full(distances.shape[0], np.inf)
    return distances.min(axis=1)


def _ray_segment(origins, directions, length, edges):
    """Distance along each ray to each segment, inf where it misses: (n_rays, n_edges)."""
    if len(edges) == 0:
        return np.full((len(origins), 0), np.inf)
    d = directions[:, None, :]
    p1 = edges[None, :, :2]
    seg = edges[None, :, 2:] - p1
    to_p1 = p1 - origins[:, None, :]

    denom = d[..., 0] * seg[..., 1] - d[..., 1] * seg[..., 0]
    parallel = np.abs(denom) < 1e-10
    safe = np.where(parallel, 1.0, denom)
    t = (to_p1[..., 0] * seg[..., 1] - to_p1[..., 1] * seg[..., 0]) / safe
    s = (to_p1[..., 0] * d[..., 1] - to_p1[..., 1] * d[..., 0]) / safe

    valid = ~parallel & (t >= 0) & (t <= length) & (s >= 0) & (s <= 1)
    return np.where(valid, t, np.inf)


def _ray_circle(origins, directions, length, centres, radius):
    """Distance along each ray to each disc, inf where it misses: (n_rays, n_discs)."""
    if len(centres) == 0:
        return np.full((len(origins), 0), np.inf)
    oc = origins[:, None, :] - centres[None, :, :]
    b = 2.0 * np.sum(oc * directions[:, None, :], axis=2)
    c = np.sum(oc ** 2, axis=2) - radius ** 2
    disc = b ** 2 - 4.0 * c
    root = np.sqrt(np.maximum(disc, 0.0))
    t1 = (-b - root) / 2.0
    t2 = (-b + root) / 2.0
    ok = disc >= 0
    t1_ok = ok & (t1 >= 0) & (t1 <= length)
    t2_ok = ok & (t2 >= 0) & (t2 <= length)
    return np.where(t1_ok, t1, np.where(t2_ok, t2, np.inf))
