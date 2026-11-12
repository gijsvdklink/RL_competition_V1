"""Train the single-agent policy with PPO (Stable-Baselines3).

The network is the one the multi-agent trainer uses (Shared/policy.py). The actor flies on
the agent's own observation (navigation-mappo-rl's encoder, then a two-layer head of
--net-width, 128 by default). The critic adds attention over the true states of the agent and
its 10 intruders, which it reads in training only. Each env runs in its own process. The
settings, KPIs and outputs are in Shared/training.py.

Run from V9_competition/:

    python -m single_agent.Training.train --name ppo_adjusted
"""

from functools import partial

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from Shared.policy import CTDEFeatures, CTDEPolicy
from Shared.training import (VALIDATION_SEED, kpi_means, make_training_env, parse_args,
                             policy_input, run_config, train)


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
                obs, _, done, truncated, info = env.step(model.predict(policy_input(model, obs), deterministic=True)[0])
            records.append(info)
        return kpi_means(records)
    return evaluate


def main():
    args = parse_args(__doc__, default_envs=8)
    cfg = run_config(args)
    factory = partial(make_training_env, make_env, args.env, cfg)
    venv = SubprocVecEnv([factory] * args.n_envs, start_method='spawn')
    venv.seed(1000 * args.seed)
    policy_kwargs = dict(
        features_extractor_class=CTDEFeatures,
        features_extractor_kwargs=dict(n_rays=cfg['n_rays'], history_length=cfg['history_length']),
        net_arch=dict(pi=[args.net_width] * 2, vf=[args.net_width] * 2),
    )
    train('sa', args, cfg, VecMonitor(venv), CTDEPolicy, policy_kwargs, validation(args.env, cfg))


if __name__ == '__main__':
    main()
