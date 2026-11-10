"""Train the single-agent policy with PPO (Stable-Baselines3).

The network is the one the multi-agent trainer uses (Shared/policy.py). The actor flies on
the agent's own observation (navigation-mappo-rl's encoder, then a two-layer head of
--net-width, 128 by default). The critic adds attention over the true states of the agent and
its 10 intruders, which it reads in training only. The
settings, KPIs and outputs are in Shared/training.py.

Run from competition-rl/:

    python -m single_agent.Training.train --steps 1e6 --name ppo
"""

from functools import partial

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from Shared.policy import CTDEFeatures, CTDEPolicy
from Shared.training import VALIDATION_SEED, kpi_means, parse_args, run_config, train


def make_env(env_kind, cfg):
    if env_kind == 'bluesky':
        from single_agent.Environment.env_bluesky import BlueSkyCompetitionEnv
        return BlueSkyCompetitionEnv(cfg)
    from single_agent.Environment.env_2d import SurrogateCompetitionEnv
    return SurrogateCompetitionEnv(cfg)


def validation(env_kind, cfg):
    env = make_env(env_kind, cfg)

    def evaluate(model, n_episodes):
        records = []
        for episode in range(n_episodes):
            obs, _ = env.reset(seed=VALIDATION_SEED if episode == 0 else None)
            done = truncated = False
            while not (done or truncated):
                obs, _, done, truncated, info = env.step(model.predict(obs, deterministic=True)[0])
            records.append(info)
        return kpi_means(records)
    return evaluate


def main():
    args = parse_args(__doc__, default_steps=1e6)
    cfg = run_config(args)
    venv = SubprocVecEnv([partial(make_env, args.env, cfg)], start_method='spawn')
    policy_kwargs = dict(
        features_extractor_class=CTDEFeatures,
        features_extractor_kwargs=dict(n_rays=cfg['n_rays'], history_length=cfg['history_length']),
        net_arch=dict(pi=[args.net_width] * 2, vf=[args.net_width] * 2),
    )
    train('sa', args, cfg, VecMonitor(venv), CTDEPolicy, policy_kwargs, validation(args.env, cfg))


if __name__ == '__main__':
    main()
