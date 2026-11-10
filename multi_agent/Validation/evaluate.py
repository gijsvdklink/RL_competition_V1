"""Score a multi-agent policy with the competition's own evaluation harness.

As single_agent/Validation/evaluate.py, for the multi-agent protocol: the first 1000
episodes of seed 42 with 10 agents. Every agent-episode is one record. Each agent acts on its
own observation through the shared actor. The critic, the only part that reads the other
aircraft's states, is not used.

    python -m multi_agent.Validation.evaluate --model Models/ma/mappo_default/best_model.zip
    python -m multi_agent.Validation.evaluate --model ... --env 2d --episodes 100
    python -m multi_agent.Validation.evaluate --episodes 50          # the neutral baseline
"""

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from Shared.harness import competition_script
from Shared.training import load_cfg
from multi_agent.Training.train import make_env


def load_policy(model_path):
    """An act(obs) -> action callable for one agent; the neutral policy without a model."""
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
    out = args.out or f'Results/ma/{run}_{args.env}_{args.episodes}ep.csv'
    harness.evaluate('ma', args.episodes, cfg['n_agents'], args.model, out)


if __name__ == '__main__':
    main()
