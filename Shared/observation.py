"""The observation: navigation-mappo-rl's state vector and lidar, plus the commanded state.

One snapshot per aircraft and decision step, 9 + 3*n_rays values:

    0  progress       (start distance - distance) / start distance; 1 at the goal
    1  cos_angle      cos of the angle from the heading to the goal
    2  speed_ratio    TAS / v_max_tas
    3  closing        speed_ratio * cos_angle
    4  distance       distance to the goal / goal_distance_norm_nm
    5  goal_cos       the goal direction in the aircraft's own frame. The repo gives a
    6  goal_sin       world-frame unit vector; the lidar is relative to the nose here, so
                      the goal is too.
    7  cmd_offset     the commanded heading as an offset from the goal bearing / 135 deg: the
                      action's own frame, so it is the heading the last action asked for
    8  cmd_speed      commanded CAS as TAS / v_max_tas: the speed once it has been reached
    9...              lidar, channel-major: obstacles, boundary, other aircraft, as in the
                      repo (Shared/lidar.py)

The policy sees history_length (4) snapshots of consecutive decisions, oldest first, as
the repo stacks its last 4 frames; Shared/encoder.py unpacks this layout. With a decision
every 10 s they span 30 s, in which an intruder moves about 2.5 NM, half the separation
minimum, about as far as an agent moved across the repo's stack. history_stride_s can space
them further, in whole decision steps.

For the centralised critic (Shared/policy.py), Observer.observe_joint adds one token per
aircraft: per agent slot in the multi-agent env, and for the agent and its 10 intruders in
the single-agent env. Only the critic reads them, in training; the actor never sees them.
Each token holds JOINT_DIM values:

    0     alive        1 while the aircraft is flying
    1     self         1 for the aircraft whose value is estimated
    2-10               the 9 state values of its newest snapshot; 0 for an intruder,
                       which has no goal
    11-12              position north, east relative to the aircraft valued / 100 NM
    13-14              velocity north, east relative to it / v_max_tas

The environments hand BlueSky's traffic arrays to the Observer as a TrafficView.
"""

from collections import deque
from typing import NamedTuple

import numpy as np
from gymnasium import spaces

from bluesky.tools.aero import vcas2tas

from core.tools import kwikqdrdist

from Shared.geo import degto180, to_local_nm
from Shared.lidar import N_CHANNELS, Lidar

STATE_DIM = 9
JOINT_DIM = 15


class TrafficView(NamedTuple):
    """The traffic arrays the observation reads; BlueSky field names."""
    id: list
    lat: np.ndarray
    lon: np.ndarray
    hdg: np.ndarray
    tas: np.ndarray
    cmd_hdg: np.ndarray
    cmd_cas: np.ndarray


def frame_dim(cfg):
    return STATE_DIM + N_CHANNELS * cfg['n_rays']


def with_self(tokens, slot):
    """The joint tokens as seen from one aircraft: it is marked as self, and every flying
    aircraft's position and velocity are made relative to its own."""
    view = tokens.copy()
    view[slot, 1] = 1.0
    flying = view[:, 0] > 0
    view[flying, 11:15] -= tokens[slot, 11:15]
    return view


class Observer:
    """Builds the stacked observation of every controlled aircraft in one scenario."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.lidar = Lidar(cfg)
        assert cfg['history_stride_s'] % cfg['action_frequency'] == 0, \
            'history_stride_s must be a whole number of decision steps'
        self.stride = cfg['history_stride_s'] // cfg['action_frequency']
        self.span = (cfg['history_length'] - 1) * self.stride + 1
        self.size = cfg['history_length'] * frame_dim(cfg)
        self._history = {}

    def reset(self, scenario, starts, goals):
        """starts and goals map each controlled aircraft to its (lat, lon)."""
        self.center = scenario.center
        sector = to_local_nm(*np.array(scenario.sector).T, self.center)
        obstacles = [to_local_nm(*np.array(o.vertices).T, self.center) for o in scenario.obstacles]
        self.lidar.set_scene(sector, obstacles)
        self.goals = goals
        self.start_dist = {ac: kwikqdrdist(*starts[ac], *goals[ac])[1] for ac in goals}
        self._history = {ac: deque(maxlen=self.span) for ac in goals}

    def space(self, n_aircraft):
        """The Dict of the actor's 'own' observation and the critic's 'joint' tokens."""
        return spaces.Dict({
            'own': spaces.Box(-np.inf, np.inf, (self.size,), np.float32),
            'joint': spaces.Box(-np.inf, np.inf, (n_aircraft, JOINT_DIM), np.float32)})

    def observe_joint(self, own_ids, slots, traffic):
        """Each own_id's observation as {'own', 'joint'}; slots lists every aircraft with a
        token, in a fixed order."""
        own = self._observe(own_ids, traffic)
        tokens = self._joint(slots, traffic)
        return {ac: {'own': own[ac], 'joint': with_self(tokens, slots.index(ac))} for ac in own_ids}

    def _observe(self, own_ids, traffic):
        """Push a snapshot for each id in own_ids and return their stacked observations."""
        rows = [traffic.id.index(ac) for ac in own_ids]
        pos_nm = to_local_nm(traffic.lat, traffic.lon, self.center)
        scans = self.lidar.scan(pos_nm[rows], traffic.hdg[rows], np.array(rows), pos_nm)
        observations = {}
        for ac, row, scan in zip(own_ids, rows, scans):
            snapshot = np.concatenate([self._state(ac, row, traffic), scan.ravel()])
            observations[ac] = self._stack(ac, snapshot.astype(np.float32))
        return observations

    def _joint(self, slots, traffic):
        """One token per slot, (len(slots), JOINT_DIM); zeros for aircraft not flying."""
        tokens = np.zeros((len(slots), JOINT_DIM), dtype=np.float32)
        pos_nm = to_local_nm(traffic.lat, traffic.lon, self.center)
        for k, ac in enumerate(slots):
            if ac not in traffic.id:
                continue
            row = traffic.id.index(ac)
            hdg = np.radians(traffic.hdg[row])
            v = traffic.tas[row] / self.cfg['v_max_tas']
            state = self._state(ac, row, traffic) if ac in self.goals else np.zeros(STATE_DIM)
            tokens[k] = [1.0, 0.0, *state, *(pos_nm[row] / 100.0), v * np.cos(hdg), v * np.sin(hdg)]
        return tokens

    def _goal(self, ac, row, traffic):
        """The goal's bearing relative to the heading, in degrees, and its distance in NM."""
        qdr, dist = kwikqdrdist(traffic.lat[row], traffic.lon[row], *self.goals[ac])
        return degto180(qdr - traffic.hdg[row]), dist

    def _state(self, ac, row, traffic):
        cfg = self.cfg
        rel_deg, dist = self._goal(ac, row, traffic)
        rel = np.radians(rel_deg)
        cos_angle = np.cos(rel)
        speed_ratio = traffic.tas[row] / cfg['v_max_tas']
        goal_bearing = traffic.hdg[row] + rel_deg
        cmd_tas = vcas2tas(traffic.cmd_cas[row], cfg['altitude_m'])
        return np.array([
            (self.start_dist[ac] - dist) / self.start_dist[ac],
            cos_angle,
            speed_ratio,
            speed_ratio * cos_angle,
            dist / cfg['goal_distance_norm_nm'],
            cos_angle,
            np.sin(rel),
            degto180(traffic.cmd_hdg[row] - goal_bearing) / cfg['max_offset_deg'],
            cmd_tas / cfg['v_max_tas'],
        ])

    def _stack(self, ac, snapshot):
        history = self._history[ac]
        history.append(snapshot)
        frames = [history[max(0, len(history) - 1 - k * self.stride)]
                  for k in reversed(range(self.cfg['history_length']))]
        return np.concatenate(frames)
