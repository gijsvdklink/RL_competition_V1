"""The single-agent competition environment in BlueSky, with our MDP on its hooks.

CompetitionEnv is subclassed as the competition intends. Overridden:
- the observation hook, and the observation space;
- the reward hook;
- action_frequency, set to our decision step (CONFIG, 5 s).

_get_action keeps the shipped HDG/SPD mapping, and the scoring layer is untouched. This is
the environment the official evaluation runs on, and the one to fine-tune in.
"""

import bluesky as bs

from bluesky_gym.envs.competition_env import CompetitionEnv

from Shared.config import CONFIG
from Shared.observation import Observer, TrafficView
from Shared.reward import gained, reward


def bluesky_view():
    traf = bs.traf
    return TrafficView(id=traf.id, lat=traf.lat, lon=traf.lon, hdg=traf.hdg, tas=traf.tas,
                       cmd_hdg=traf.ap.trk, cmd_cas=traf.selspd)


class BlueSkyCompetitionEnv(CompetitionEnv):

    def __init__(self, cfg=CONFIG, render_mode=None, scenario=None):
        super().__init__(render_mode=render_mode, scenario=scenario,
                         action_frequency=cfg['action_frequency'])
        self.cfg = cfg
        self.observer = Observer(cfg)
        self.observation_space = self.observer.space(1 + self.n_intruders)
        self._new_episode = False

    def reset(self, seed=None, options=None):
        self._new_episode = True
        return super().reset(seed=seed, options=options)

    def _get_obs(self, ac_id):
        if self._new_episode:
            spec = self.scenario.agents[0]
            self.observer.reset(self.scenario, {ac_id: spec.start}, {ac_id: spec.goal})
            self._new_episode = False
        slots = [ac_id] + self.intruder_ids
        return self.observer.observe_joint([ac_id], slots, bluesky_view())[ac_id]

    def _get_reward(self, ac_id):
        return reward(self.cfg['reward'], gained(self.metrics[ac_id], self._reward_baseline))
