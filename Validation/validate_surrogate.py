"""Fly the same episodes in the surrogate and in BlueSky, and report how far they drift apart.

A seed gives the same scenario in both environments. Each seed is flown with the same
scripted actions in both:

    neutral   [0, 0]: keep heading and speed, the competition's baseline policy
    random    uniform actions from a fixed generator, a new one every second
    weave     +-45 deg heading changes held for 60 s, with speed steps

For each seed and policy the script reports:
- the agent's position error at fixed times;
- the largest intruder position error;
- the difference in every scored metric.

The neutral rows also give both environments' mean metrics, which is what the surrogate
must get right for training to transfer. Run from competition-rl/:

    python -m Validation.validate_surrogate --seeds 20
"""

import argparse

import bluesky as bs
import numpy as np

from core.tools import kwikqdrdist

from Shared.config import CONFIG
from Shared.metrics import METRIC_KEYS
from single_agent.Environment.env_2d import SurrogateCompetitionEnv
from single_agent.Environment.env_bluesky import BlueSkyCompetitionEnv

CHECK_TIMES = (60, 300, 600)
CFG = {**CONFIG, 'action_frequency': 1}      # compare the dynamics second by second
COMPARED = [k for k in METRIC_KEYS if k != 'total_reward'] + ['total_reward']


def neutral(t, rng):
    return np.zeros(2)


def random_actions(t, rng):
    return rng.uniform(-1.0, 1.0, 2)


def weave(t, rng):
    phase = (t // 60) % 4
    return np.array([[1.0, 0.0, -1.0, 0.0][phase], [0.0, 1.0, 0.0, -1.0][phase]])


POLICIES = {'neutral': neutral, 'random': random_actions, 'weave': weave}


def surrogate_positions(env):
    t = env.traffic
    return np.stack([t.lat, t.lon], axis=1), list(t.id)


def bluesky_positions(env):
    return np.stack([bs.traf.lat, bs.traf.lon], axis=1), list(bs.traf.id)


def fly(env, positions, seed, policy):
    """One episode; returns the final info and every aircraft's track by id."""
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    tracks, info, t = {}, {}, 0
    done = truncated = False
    while not (done or truncated):
        _, _, done, truncated, info = env.step(policy(t, rng))
        t += 1
        latlon, ids = positions(env)
        for ac, p in zip(ids, latlon):
            tracks.setdefault(ac, []).append(p)
    return info, {ac: np.array(p) for ac, p in tracks.items()}


def separation_nm(a, b):
    n = min(len(a), len(b))
    _, d = kwikqdrdist(a[:n, 0], a[:n, 1], b[:n, 0], b[:n, 1])
    return d


def compare(seeds, policy_name):
    surrogate, blue = SurrogateCompetitionEnv(CFG), BlueSkyCompetitionEnv(CFG)
    policy = POLICIES[policy_name]
    rows = []
    for seed in seeds:
        info_s, tracks_s = fly(surrogate, surrogate_positions, seed, policy)
        info_b, tracks_b = fly(blue, bluesky_positions, seed, policy)
        agent = surrogate.agent
        err = separation_nm(tracks_s[agent], tracks_b[agent])
        intruders = max(separation_nm(tracks_s[ac], tracks_b[ac]).max() for ac in surrogate.intruder_ids)
        at = [err[t - 1] if len(err) >= t else np.nan for t in CHECK_TIMES]
        rows.append((seed, at, intruders, info_s, info_b))
    return rows


def report(policy_name, rows):
    print(f'\n### {policy_name}\n')
    head = ' '.join(f'err@{t}s' for t in CHECK_TIMES)
    print(f'seed  {head}  intr.max  metric differences (surrogate - BlueSky)')
    for seed, at, intruders, info_s, info_b in rows:
        diffs = {k: info_s[k] - info_b[k] for k in COMPARED[:-1] if info_s[k] != info_b[k]}
        errs = ' '.join(f'{e:7.3f}' for e in at)
        print(f'{seed:>4}  {errs}  {intruders:8.3f}  {diffs or "identical"}')
    print('\nmean over seeds      surrogate     BlueSky')
    for key in COMPARED:
        s = np.mean([r[3][key] for r in rows])
        b = np.mean([r[4][key] for r in rows])
        print(f'{key:<24} {s:>10.2f}  {b:>10.2f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--policies', nargs='+', default=list(POLICIES), choices=list(POLICIES))
    args = parser.parse_args()
    seeds = range(1000, 1000 + args.seeds)
    for name in args.policies:
        report(name, compare(seeds, name))


if __name__ == '__main__':
    main()
