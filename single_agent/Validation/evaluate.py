"""Score a single-agent policy with the competition's own evaluation harness.

scripts/evaluate_competition.py (in ../bluesky-gym) holds the official protocol:
- the first 1000 episodes of seed 42;
- its metric table;
- its per-episode CSV.

This script imports it and swaps in its two edit points, make_env and load_policy, so the
official file stays untouched. The observation settings come from the run's config.json,
and the episodes follow the competition's rules: no termination on a violation.

    python -m single_agent.Validation.evaluate --model Models/sa/ppo_default/best_model.zip
    python -m single_agent.Validation.evaluate --model ... --episodes 100
    python -m single_agent.Validation.evaluate --episodes 50          # the neutral baseline
"""

import argparse
from pathlib import Path

from Shared.harness import competition_script
from Shared.training import evaluation_cfg, load_cfg, load_policy
from single_agent.Training.train import make_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None)
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    harness = competition_script('evaluate_competition')
    cfg = evaluation_cfg(load_cfg(args.model))
    act = load_policy(args.model)
    harness.make_env = lambda kind, n_agents=None: make_env(cfg)
    harness.load_policy = lambda kind, model_path=None: act

    run = Path(args.model).parent.name if args.model else 'neutral'
    out = args.out or f'Results/sa/{run}_{args.episodes}ep.csv'
    harness.evaluate('sa', args.episodes, harness.N_AGENTS_MA, args.model, out)


if __name__ == '__main__':
    main()
