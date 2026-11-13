"""Train the multi-agent policy: MAPPO as Stable-Baselines3 PPO with parameter sharing.

All 10 aircraft share one decentralised actor and one centralised critic (Shared/policy.py).
Every aircraft is a sub-env of ZooVecEnv (multi_agent/Training/vec_env.py), so --steps counts
aircraft decisions; 8 env copies x 10 aircraft x 12 decisions fill the buffer of 960.
The settings, KPIs and outputs are in Shared/training.py.

Run from RL_COMPETITION_V2/:

    python -m multi_agent.Training.train --steps 5e6 --name mappo
"""

from functools import partial

import numpy as np

from Shared.policy import CTDEFeatures, CTDEPolicy
from Shared.config import CONFIG
from Shared.training import VALIDATION_SEED, evaluation_cfg, kpi_means, parse_args, train
from multi_agent.Training.vec_env import ZooVecEnv


def make_env(cfg):
    from multi_agent.Environment.env_bluesky import BlueSkyCompetitionZooEnv
    return BlueSkyCompetitionZooEnv(cfg)


def act(model, observations, agents):
    """One deterministic action per agent; each from its own observation."""
    batch = {key: np.stack([observations[a][key] for a in agents]) for key in ('own', 'joint')}
    batch = model.get_vec_normalize_env().normalize_obs(batch)
    return dict(zip(agents, model.predict(batch, deterministic=True)[0]))


def validation(cfg):
    env = make_env(evaluation_cfg(cfg))

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
    args = parse_args(__doc__, default_steps=5e6)
    cfg = CONFIG
    venv = ZooVecEnv(partial(make_env, cfg), args.n_envs)
    policy_kwargs = dict(
        features_extractor_class=CTDEFeatures,
        features_extractor_kwargs=dict(n_rays=cfg['n_rays'], history_length=cfg['history_length']),
        net_arch=dict(pi=[args.net_width] * 2, vf=[args.net_width] * 2),
    )
    train('ma', args, cfg, venv, CTDEPolicy, policy_kwargs, validation(cfg))


if __name__ == '__main__':
    main()
