"""The multi-agent competition environment on the 2D surrogate instead of BlueSky.

This is CompetitionZooEnv (bluesky_zoo/competition/competition.py) with BlueSky replaced by
Shared/traffic.py, and our MDP. It is a PettingZoo ParallelEnv with the competition's
agent lifecycle:
- an agent terminates when it captures its goal; its final observation, reward and info
  come in that step, then its aircraft is removed so it no longer conflicts with anyone;
- the remaining agents get the full step;
- everyone truncates when the shared clock reaches 3000 s.

Intrusions count in both aircraft's metrics, as in the competition. The scenario comes from
core.scenario.ScenarioGenerator with the same generator calls, so a seed gives the same
scenario as in BlueSky. Each observation is a dict:
- 'own', the aircraft's own observation, which the actor acts on;
- 'joint', every aircraft's state, which only the centralised critic reads.
"""

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv

from core.scenario import ScenarioGenerator, agent_callsigns

from Shared.actions import apply_action
from Shared.config import CONFIG
from Shared.metrics import Scoring
from Shared.observation import Observer
from Shared.reward import team_reward
from Shared.traffic import Traffic


class SurrogateCompetitionZooEnv(ParallelEnv):

    metadata = {'name': 'competition_surrogate_v0', 'render_modes': ['human', 'rgb_array'],
                'render_fps': 30}

    def __init__(self, cfg=CONFIG, n_agents=None, render_mode=None):
        self.cfg = cfg
        self.render_mode = render_mode
        self.n_agents = n_agents or cfg['n_agents']
        self.possible_agents = agent_callsigns(self.n_agents)
        self.agents = []
        self.traffic = Traffic(cfg['altitude_m'])
        self.observer = Observer(cfg)
        self.scenario_generator = ScenarioGenerator(
            n_obstacles=cfg['n_obstacles'], n_intruders=0, n_agents=self.n_agents,
            agent_speed=cfg['ac_spd'], center=cfg['center'])
        observation_space = self.observer.space(self.n_agents)
        action_space = spaces.Box(-1.0, 1.0, (2,), np.float64)
        self.observation_spaces = {a: observation_space for a in self.possible_agents}
        self.action_spaces = {a: action_space for a in self.possible_agents}
        self._np_random = None

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None or self._np_random is None:
            self._np_random, _ = seeding.np_random(seed)
        self.scenario = self.scenario_generator.generate(self._np_random)

        self.traffic.reset()
        self.agents = list(self.possible_agents)
        self.goals, starts = {}, {}
        for spec in self.scenario.agents:
            self.traffic.create(spec.ac_id, *spec.start, spec.heading, spec.speed)
            self.goals[spec.ac_id], starts[spec.ac_id] = tuple(spec.goal), tuple(spec.start)

        self.sim_time = 0.0
        self.scoring = Scoring(self.scenario, self.cfg['intrusion_distance_nm'],
                               self.cfg['distance_margin_km'])
        for agent in self.agents:
            self.scoring.add(agent)
            self._update_metrics(agent, dt=0.0)
        self.observer.reset(self.scenario, starts, self.goals)

        observations = self._observe(self.agents)
        return observations, {a: self._get_info(a) for a in self.agents}

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}
        for agent in self.agents:
            apply_action(self.traffic, agent, actions[agent], self.cfg)
        before = {a: dict(self.scoring.metrics[a]) for a in self.agents}

        for _ in range(self.cfg['action_frequency']):
            if not self.agents:
                break
            self.traffic.step(1.0)
            self.sim_time += 1.0
            for agent in self.agents:
                self._update_metrics(agent)
            reached = [a for a in self.agents if self.scoring.waypoint_reached[a]]
            if reached:
                observations.update(self._observe(reached))
            for agent in reached:
                rewards[agent] = self._reward(agent, before)
                terminations[agent], truncations[agent] = True, False
                infos[agent] = self._get_info(agent)
                self.traffic.delete(agent)
                self.agents.remove(agent)

        truncate_all = self.sim_time >= self.cfg['episode_time_limit_s']
        if self.agents:
            observations.update(self._observe(self.agents))
        for agent in self.agents:
            rewards[agent] = self._reward(agent, before)
            terminations[agent], truncations[agent] = False, truncate_all
            infos[agent] = self._get_info(agent)
        if truncate_all:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def _observe(self, agents):
        return self.observer.observe_joint(agents, self.possible_agents, self.traffic)

    def _reward(self, agent, before):
        r = team_reward(self.cfg['reward'], self.scoring.metrics, before, agent)
        self.scoring.metrics[agent]['total_reward'] += r
        return r

    def _update_metrics(self, agent, dt=1.0):
        t = self.traffic
        i = t.id2idx(agent)
        others = {a: (t.lat[t.id2idx(a)], t.lon[t.id2idx(a)]) for a in self.agents if a != agent}
        self.scoring.update(agent, t.lat[i], t.lon[i], self.goals[agent], others, dt)

    def _get_info(self, agent):
        return dict(self.scoring.metrics[agent])
