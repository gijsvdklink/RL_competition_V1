"""The single-agent competition environment on the 2D surrogate instead of BlueSky.

This is CompetitionEnv (bluesky_gym/envs/competition_env.py) with two changes:
- BlueSky is replaced by Shared/traffic.py;
- the MDP is ours: the lidar observation, the V8 reward, a decision every 5 s.

Everything else matches the competition, so the same seed gives the same episode, up to the
surrogate's error:

    scenario  core.scenario.ScenarioGenerator, drawn from self.np_random as in reset()
    aircraft  the agent, and 10 intruders on their ADDWPT routes, created as in reset()
    scoring   Shared/metrics.py, a line-for-line copy of _update_metrics
    actions   HDG and SPD relative to the current state, as core/actions.py

It terminates when the goal is captured and truncates at 3000 simulated seconds.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from core.scenario import ScenarioGenerator

from Shared.actions import apply_action
from Shared.config import CONFIG
from Shared.geo import initial_bearing
from Shared.metrics import Scoring
from Shared.observation import Observer
from Shared.reward import gained, reward
from Shared.traffic import Traffic


class SurrogateCompetitionEnv(gym.Env):

    metadata = {'render_modes': ['rgb_array', 'human'], 'render_fps': 30}

    def __init__(self, cfg=CONFIG, render_mode=None):
        self.cfg = cfg
        self.render_mode = render_mode
        self.traffic = Traffic(cfg['altitude_m'])
        self.observer = Observer(cfg)
        self.scenario_generator = ScenarioGenerator(
            n_obstacles=cfg['n_obstacles'], n_intruders=cfg['n_intruders'],
            agent_speed=cfg['ac_spd'], center=cfg['center'])
        self.observation_space = self.observer.space(1 + cfg['n_intruders'])
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float64)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.scenario = self.scenario_generator.generate(self.np_random)
        spec = self.scenario.agents[0]
        self.agent, self.goal = spec.ac_id, tuple(spec.goal)

        self.traffic.reset()
        self.traffic.create(self.agent, *spec.start, spec.heading, spec.speed)
        self.intruder_ids = []
        for route in self.scenario.intruder_routes:
            (lat, lon), (next_lat, next_lon) = route.waypoints[:2]
            self.traffic.create(route.ac_id, lat, lon,
                                initial_bearing(lat, lon, next_lat, next_lon), route.speed)
            self.traffic.add_route(route.ac_id, route.waypoints[1:])
            self.intruder_ids.append(route.ac_id)

        self.scoring = Scoring(self.scenario, self.cfg['intrusion_distance_nm'],
                               self.cfg['distance_margin_km'])
        self.scoring.add(self.agent)
        self._update_metrics(dt=0.0)
        self.observer.reset(self.scenario, {self.agent: spec.start}, {self.agent: self.goal})
        return self._get_obs(), self._get_info()

    def step(self, action):
        apply_action(self.traffic, self.agent, action, self.cfg)
        before = dict(self.metrics)
        for _ in range(self.cfg['action_frequency']):
            self.traffic.step(1.0)
            self._update_metrics()
            if self.scoring.waypoint_reached[self.agent]:
                break

        r = reward(self.cfg['reward'], gained(self.metrics, before))
        self.metrics['total_reward'] += r
        terminated = self.scoring.waypoint_reached[self.agent]
        truncated = self.metrics['flight_time'] >= self.cfg['episode_time_limit_s']
        return self._get_obs(), r, terminated, truncated, self._get_info()

    @property
    def metrics(self):
        return self.scoring.metrics[self.agent]

    def _update_metrics(self, dt=1.0):
        t = self.traffic
        i = t.id2idx(self.agent)
        others = {ac: (t.lat[j], t.lon[j]) for ac in self.intruder_ids if (j := t.id2idx(ac)) >= 0}
        self.scoring.update(self.agent, t.lat[i], t.lon[i], self.goal, others, dt)

    def _get_obs(self):
        slots = [self.agent] + self.intruder_ids
        return self.observer.observe_joint([self.agent], slots, self.traffic)[self.agent]

    def _get_info(self):
        return dict(self.metrics)
