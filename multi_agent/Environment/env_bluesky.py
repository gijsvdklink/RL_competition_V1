"""The multi-agent competition environment in BlueSky, with our MDP on its hooks.

CompetitionZooEnv is subclassed as the competition intends. Overridden:
- the observation hook, and the observation spaces;
- the reward hook;
- action_frequency, set to our decision step (CONFIG, 5 s).

_get_action keeps the shipped HDG/SPD mapping, and the scoring layer is untouched.
BlueSky is a process-wide singleton, so each process can hold only one of these.
"""

import bluesky as bs

from bluesky_zoo.competition.competition import CompetitionZooEnv

from Shared.config import CONFIG
from Shared.observation import Observer
from Shared.reward import drift, reward
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

    def _get_obs(self, ac_id):
        if self._new_episode:
            starts = {spec.ac_id: spec.start for spec in self.scenario.agents}
            self.observer.reset(self.scenario, starts, self._goal)
            self._new_episode = False
        return self.observer.observe_joint([ac_id], self.possible_agents, bluesky_view())[ac_id]

    def _get_reward(self, ac_id):
        i = bs.traf.id2idx(ac_id)
        drift_rad = drift(bs.traf.lat[i], bs.traf.lon[i], bs.traf.hdg[i], self._goal[ac_id])
        return reward(self.metrics[ac_id], self._reward_baseline[ac_id], drift_rad, self.cfg)
