"""The single-agent competition environment in BlueSky, with our MDP on its hooks.

CompetitionEnv is subclassed as the competition intends. Overridden:
- the observation hook, and the observation space (Shared/observation.py);
- the action hook: the goal-relative mapping of Shared/actions.py;
- the reward hook (Shared/reward.py);
- action_frequency, set to our decision step (10 s).

The scoring layer is untouched. With terminate_on_violation (training), a step that brings a
new loss of separation, restricted-area entry or sector exit also ends the episode, as a
collision does in navigation-mappo-rl; info then carries ended_by_violation. With the flag
off (evaluation, the official harness) the episode follows the competition's own rules.
"""

import bluesky as bs

from bluesky_gym.envs.competition_env import CompetitionEnv

from Shared.actions import stack_commands
from Shared.config import CONFIG
from Shared.observation import Observer, TrafficView
from Shared.reward import drift, new_violation, reward


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

    def step(self, action):
        observation, r, terminated, truncated, info = super().step(action)
        crashed = (self.cfg['terminate_on_violation']
                   and new_violation(self.metrics[self.agent], self._reward_baseline))
        return (observation, r, terminated or crashed, truncated and not crashed,
                {**info, 'ended_by_violation': crashed})

    def _get_obs(self, ac_id):
        if self._new_episode:
            spec = self.scenario.agents[0]
            self.observer.reset(self.scenario, {ac_id: spec.start}, {ac_id: spec.goal})
            self._new_episode = False
        slots = [ac_id] + self.intruder_ids
        return self.observer.observe_joint([ac_id], slots, bluesky_view())[ac_id]

    def _get_action(self, ac_id, action):
        stack_commands(bs, ac_id, (self.goal_lat, self.goal_lon), action, self.cfg)

    def _get_reward(self, ac_id):
        i = bs.traf.id2idx(ac_id)
        drift_rad = drift(bs.traf.lat[i], bs.traf.lon[i], bs.traf.hdg[i], (self.goal_lat, self.goal_lon))
        return reward(self.metrics[ac_id], self._reward_baseline, drift_rad, self.cfg)
