"""Score a single-agent policy with the competition's own evaluation harness.

scripts/evaluate_competition.py (in ../bluesky-gym) holds the official protocol:
- the first 1000 episodes of seed 42;
- its metric table;
- its per-episode CSV.

This script imports it and swaps in its two edit points, make_env and load_policy, so the
official file stays untouched. The observation settings come from the run's config.json.
Only BlueSky numbers are reportable; --env 2d is a quick check on the surrogate.

    python -m single_agent.Validation.evaluate --model Models/sa/ppo_default/best_model.zip
    python -m single_agent.Validation.evaluate --model ... --env 2d --episodes 100
    python -m single_agent.Validation.evaluate --episodes 50          # the neutral baseline
"""

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from Shared.harness import competition_script
from Shared.training import load_cfg
from single_agent.Training.train import make_env


def load_policy(model_path):
    """An act(obs) -> action callable; the neutral fly-straight policy without a model."""
    if model_path is None:
        return lambda obs: np.zeros(2)
    model = PPO.load(model_path, device='cpu')
    return lambda obs: model.predict(obs, deterministic=True)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None)
    parser.add_argument('--env', choices=['2d', 'bluesky'], default='bluesky')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    harness = competition_script('evaluate_competition')
    cfg = load_cfg(args.model)
    act = load_policy(args.model)
    harness.make_env = lambda kind, n_agents=None: make_env(args.env, cfg)
    harness.load_policy = lambda kind, model_path=None: act

    run = Path(args.model).parent.name if args.model else 'neutral'
    out = args.out or f'Results/sa/{run}_{args.env}_{args.episodes}ep.csv'
    harness.evaluate('sa', args.episodes, harness.N_AGENTS_MA, args.model, out)


if __name__ == '__main__':
    main()
