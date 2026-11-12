"""The multi-agent env as a Stable-Baselines3 VecEnv: one sub-env per (env copy, agent slot).

SB3's PPO then trains one shared policy on every aircraft (parameter sharing). Each env copy
runs in its own process, which also gives each BlueSky env the singleton it needs.

This replaces SuperSuit's chain (black_death_v3 -> pettingzoo_env_to_vec_env_v1). In
SuperSuit 3.11, black_death reports an env as done only when every agent is both terminated
and truncated. It also hides which agent finished, and it drops the observation at
truncation. Here every slot keeps its own episode:

- **while flying,** a slot returns its own observation and reward;
- **when it leaves** (goal captured or time up), it returns done with the competition metrics
  in info, plus 'episode', 'terminal_observation' and 'TimeLimit.truncated', so SB3 logs the
  episode and bootstraps a truncation from its last observation;
- **after it has left,** until the whole env resets, it returns a zero observation, reward 0
  and done on every step. Each such step is then an episode of its own that SB3 does not log
  and whose value target is 0. SB3 has no sample mask, so these steps still enter the batch.
  Their advantage is near zero, so their effect on the policy is small.

An env whose last agent has left resets at once.
"""

import multiprocessing as mp

import numpy as np
from stable_baselines3.common.vec_env import VecEnv


class ZooVecEnv(VecEnv):

    def __init__(self, make_env, n_envs, seed=0):
        ctx = mp.get_context('spawn')
        self.remotes, work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = [ctx.Process(target=_worker, args=(remote, make_env), daemon=True)
                          for remote in work_remotes]
        for p in self.processes:
            p.start()
        for remote in work_remotes:
            remote.close()
        self.remotes[0].send(('spaces', None))
        self.n_agents, observation_space, action_space = self.remotes[0].recv()
        self.n_envs, self.seed_base = n_envs, seed
        super().__init__(n_envs * self.n_agents, observation_space, action_space)

    def reset(self):
        for i, remote in enumerate(self.remotes):
            remote.send(('reset', self.seed_base + i))
        return _stack([obs for remote in self.remotes for obs in remote.recv()])

    def step_async(self, actions):
        for i, remote in enumerate(self.remotes):
            remote.send(('step', actions[i * self.n_agents:(i + 1) * self.n_agents]))

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        obs = [o for r in results for o in r[0]]
        rewards = np.concatenate([r[1] for r in results]).astype(np.float32)
        dones = np.concatenate([r[2] for r in results])
        infos = [info for r in results for info in r[3]]
        return _stack(obs), rewards, dones, infos

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.processes:
            p.join()

    def get_attr(self, attr_name, indices=None):
        if attr_name == 'render_mode':
            return [None] * len(self._get_indices(indices))
        raise AttributeError(attr_name)

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._get_indices(indices))


def _stack(observations):
    return {key: np.stack([o[key] for o in observations]) for key in observations[0]}


def _worker(remote, make_env):
    env = make_env()
    slots = env.possible_agents
    space = env.observation_space(slots[0])
    blank = {key: np.zeros(sub.shape, dtype=np.float32) for key, sub in space.spaces.items()}
    returns = dict.fromkeys(slots, 0.0)
    lengths = dict.fromkeys(slots, 0)

    def pack(observations):
        return [observations.get(agent, blank) for agent in slots]

    while True:
        cmd, data = remote.recv()
        if cmd == 'spaces':
            remote.send((len(slots), space, env.action_space(slots[0])))
        elif cmd == 'reset':
            observations, _ = env.reset(seed=data)
            returns.update(dict.fromkeys(slots, 0.0))
            lengths.update(dict.fromkeys(slots, 0))
            remote.send(pack(observations))
        elif cmd == 'step':
            remote.send(_step(env, slots, data, pack, returns, lengths))
        elif cmd == 'close':
            env.close()
            remote.close()
            break


def _step(env, slots, actions, pack, returns, lengths):
    flying = [agent in env.agents for agent in slots]
    observations, rewards, terms, truncs, infos = env.step(
        {agent: actions[k] for k, agent in enumerate(slots) if flying[k]})

    reward = np.zeros(len(slots))
    done = np.ones(len(slots), dtype=bool)
    out_infos = [{} for _ in slots]
    for k, agent in enumerate(slots):
        if not flying[k]:
            continue
        reward[k] = rewards[agent]
        returns[agent] += rewards[agent]
        lengths[agent] += 1
        done[k] = terms[agent] or truncs[agent]
        if done[k]:
            out_infos[k] = {**infos[agent], 'terminal_observation': observations[agent],
                            'TimeLimit.truncated': bool(truncs[agent] and not terms[agent]),
                            'episode': {'r': returns[agent], 'l': lengths[agent], 't': 0.0}}

    if env.agents:
        next_obs = pack({a: observations[a] for a in env.agents})
    else:
        next_obs = pack(env.reset()[0])
        returns.update(dict.fromkeys(slots, 0.0))
        lengths.update(dict.fromkeys(slots, 0))
    return next_obs, reward, done, out_infos
