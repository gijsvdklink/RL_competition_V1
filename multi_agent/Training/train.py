"""Train the multi-agent policy: MAPPO as Stable-Baselines3 PPO with parameter sharing.

All 10 aircraft share one decentralised actor and one centralised critic
(multi_agent/Training/policy.py). Every aircraft of every env copy is a sub-env of
ZooVecEnv (multi_agent/Training/vec_env.py), so --steps counts aircraft decisions, and
n_steps is the rollout per aircraft slot. With 4 env copies that is 40 slots, a buffer of
40,000 samples per rollout. The settings, KPIs and outputs are in Shared/training.py.

Run from competition-rl/:

    python -m multi_agent.Training.train --steps 5e6 --name mappo_default
    python -m multi_agent.Training.train --steps 5e6 --n-epochs 30 --batch-size full --name mappo_adjusted
"""

from functools import partial

import numpy as np

from Shared.training import VALIDATION_SEED, kpi_means, parse_args, run_config, train
from multi_agent.Training.policy import CTDEFeatures, CTDEPolicy
from multi_agent.Training.vec_env import ZooVecEnv


def make_env(env_kind, cfg):
    if env_kind == 'bluesky':
        from multi_agent.Environment.env_bluesky import BlueSkyCompetitionZooEnv
        return BlueSkyCompetitionZooEnv(cfg)
    from multi_agent.Environment.env_2d import SurrogateCompetitionZooEnv
    return SurrogateCompetitionZooEnv(cfg)


def act(model, observations, agents):
    """One deterministic action per agent; each from its own observation."""
    batch = {key: np.stack([observations[a][key] for a in agents]) for key in ('own', 'joint')}
    return dict(zip(agents, model.predict(batch, deterministic=True)[0]))


def validation(env_kind, cfg):
    env = make_env(env_kind, cfg)

    def evaluate(model, n_episodes):
        records, team = [], []
        for episode in range(n_episodes):
            observations, _ = env.reset(seed=VALIDATION_SEED if episode == 0 else None)
            finished = []
            while env.agents:
                observations, _, terms, truncs, infos = env.step(act(model, observations, list(env.agents)))
                finished += [infos[a] for a in infos if terms[a] or truncs[a]]
            records += finished
            team.append(sum(r['total_reward'] for r in finished))
        return {**kpi_means(records), 'team_total_reward': float(np.mean(team))}
    return evaluate


def main():
    args = parse_args(__doc__, default_steps=5e6, default_envs=4)
    cfg = run_config(args)
    venv = ZooVecEnv(partial(make_env, args.env, cfg), args.n_envs, seed=1000 * args.seed)
    policy_kwargs = dict(
        features_extractor_class=CTDEFeatures,
        features_extractor_kwargs=dict(n_rays=cfg['n_rays'], history_length=cfg['history_length']),
        net_arch=dict(pi=[args.net_width] * 2, vf=[args.net_width] * 2),
    )
    train('ma', args, cfg, venv, CTDEPolicy, policy_kwargs, validation(args.env, cfg))


if __name__ == '__main__':
    main()
