"""Train the single-agent policy with PPO (Stable-Baselines3).

The network is the one the multi-agent trainer uses (Shared/policy.py). The actor flies on
the agent's own observation (navigation-mappo-rl's encoder, then a two-layer head of
--net-width, 128 by default). The critic adds attention over the true states of the agent and
its 10 intruders, which it reads in training only. The settings, KPIs and outputs are in
Shared/training.py.

Run from RL_COMPETITION_V2/:

    python -m single_agent.Training.train --steps 1e6 --name ppo
"""

from functools import partial

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from Shared.policy import CTDEFeatures, CTDEPolicy
from Shared.config import CONFIG
from Shared.training import VALIDATION_SEED, evaluation_cfg, kpi_means, parse_args, train


def make_env(cfg):
    from single_agent.Environment.env_bluesky import BlueSkyCompetitionEnv
    return BlueSkyCompetitionEnv(cfg)


def validation(cfg):
    env = make_env(evaluation_cfg(cfg))

    def evaluate(model, n_episodes):
        records = []
        for episode in range(n_episodes):
            obs, _ = env.reset(seed=VALIDATION_SEED if episode == 0 else None)
            done = truncated = False
            while not (done or truncated):
                action = model.predict(model.get_vec_normalize_env().normalize_obs(obs), deterministic=True)[0]
                obs, _, done, truncated, info = env.step(action)
            records.append(info)
        return kpi_means(records)
    return evaluate


def main():
    args = parse_args(__doc__, default_steps=1e6)
    cfg = CONFIG
    venv = SubprocVecEnv([partial(make_env, cfg) for _ in range(args.n_envs)], start_method='spawn')
    policy_kwargs = dict(
        features_extractor_class=CTDEFeatures,
        features_extractor_kwargs=dict(n_rays=cfg['n_rays'], history_length=cfg['history_length']),
        net_arch=dict(pi=[args.net_width] * 2, vf=[args.net_width] * 2),
    )
    train('sa', args, cfg, VecMonitor(venv), CTDEPolicy, policy_kwargs, validation(cfg))


if __name__ == '__main__':
    main()
