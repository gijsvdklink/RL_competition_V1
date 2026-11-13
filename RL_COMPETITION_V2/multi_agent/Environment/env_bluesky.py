"""The multi-agent competition environment in BlueSky, with our MDP on its hooks.

CompetitionZooEnv is subclassed as the competition intends. Overridden:
- the observation hook, and the observation spaces (Shared/observation.py);
- the action hook: the goal-relative mapping of Shared/actions.py;
- the reward hook (Shared/reward.py);
- action_frequency, set to our decision step (10 s).

The scoring layer is untouched. With terminate_on_violation (training), an agent whose step
brings a new loss of separation, restricted-area entry or sector exit also leaves the
episode, and its aircraft is removed, as a collision does in navigation-mappo-rl. A loss of
separation counts for both aircraft, so both leave. With the flag off (evaluation, the
official harness) the env follows the competition's own rules. BlueSky is a process-wide
singleton, so each process can hold only one of these.
"""

import bluesky as bs

from bluesky_zoo.competition.competition import CompetitionZooEnv

from Shared.actions import stack_commands
from Shared.config import CONFIG
from Shared.observation import Observer
from Shared.reward import drift, new_violation, reward
from single_agent.Environment.env_bluesky import bluesky_view


class BlueSkyCompetitionZooEnv(CompetitionZooEnv):

    def __init__(self, cfg=CONFIG, n_agents=None, render_mode=None, scenario=None):
        super().__init__(render_mode=render_mode, scenario=scenario,
                         n_agents=n_agents or cfg['n_agents'],
                         action_frequency=cfg['action_frequency'])
        self.cfg = cfg
        self.observer = Observer(cfg)
        space = self.observer.space(self.n_agents)
        self.observation_spaces = {a: space for a in self.possible_agents}
        self._new_episode = False

    def reset(self, seed=None, options=None):
        self._new_episode = True
        return super().reset(seed=seed, options=options)

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = super().step(actions)
        infos = {a: {**info, 'ended_by_violation': False} for a, info in infos.items()}
        if self.cfg['terminate_on_violation']:
            crashed = [a for a in self.agents if new_violation(self.metrics[a], self._reward_baseline[a])]
            for agent in crashed:
                terminations[agent], truncations[agent] = True, False
                infos[agent]['ended_by_violation'] = True
                bs.traf.delete(bs.traf.id2idx(agent))
                self.agents.remove(agent)
        return observations, rewards, terminations, truncations, infos

    def _get_obs(self, ac_id):
        if self._new_episode:
            starts = {spec.ac_id: spec.start for spec in self.scenario.agents}
            self.observer.reset(self.scenario, starts, self._goal)
            self._new_episode = False
        return self.observer.observe_joint([ac_id], self.possible_agents, bluesky_view())[ac_id]

    def _get_action(self, ac_id, action):
        stack_commands(bs, ac_id, self._goal[ac_id], action, self.cfg)

    def _get_reward(self, ac_id):
        i = bs.traf.id2idx(ac_id)
        drift_rad = drift(bs.traf.lat[i], bs.traf.lon[i], bs.traf.hdg[i], self._goal[ac_id])
        return reward(self.metrics[ac_id], self._reward_baseline[ac_id], drift_rad, self.cfg)
