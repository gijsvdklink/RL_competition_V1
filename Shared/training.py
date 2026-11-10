"""What both trainers share: the PPO settings, the command line, and the TensorBoard KPIs.

Both train with Stable-Baselines3 PPO, on the surrogate by default (--env 2d). VecNormalize
scales the reward only: the observation is normalised by construction, and the -100
loss-of-separation penalty would otherwise dominate the value loss.

Update schedules, both with a rollout of 1000 decision steps per env:

    default   10 epochs, minibatches of 512
    adjusted  30 epochs, the whole rollout buffer as one batch (--batch-size full)

TensorBoard, under Models/<sa|ma>/<run>/:

    rollout/ep_rew_mean  the total reward per (agent-)episode, SB3's own, over the last 100
    rollout/<kpi>        the goal completion rate and the nine scored metrics, averaged over
                         the training episodes that ended in that rollout
    eval/*               every eval_every steps, eval_episodes deterministic episodes on
                         held-out scenarios (seed 7, never the competition's 42): the goal
                         completion rate, the nine metrics and, for multi-agent, the
                         team's total reward

The model with the best eval/total_reward is kept as best_model.zip.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

from Shared.config import CONFIG
from Shared.metrics import METRIC_KEYS

SETTINGS = {
    'learning_rate':   3e-4,
    'n_steps':         1000,       # decision steps per env per rollout
    'batch_size':      512,
    'n_epochs':        10,
    'gamma':           0.99,
    'gae_lambda':      0.95,
    'clip_range':      0.2,
    'ent_coef':        0.0,
    'vf_coef':         0.5,
    'max_grad_norm':   0.5,
}

SAVE_EVERY      = 500_000          # steps
EVAL_EVERY      = 100_000          # steps
EVAL_EPISODES   = 20
VALIDATION_SEED = 7


def parse_args(doc, default_steps, default_envs):
    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--env', choices=['2d', 'bluesky'], default='2d')
    parser.add_argument('--decision-step', type=int, default=CONFIG['action_frequency'], help='seconds')
    parser.add_argument('--history-stride', type=int, default=CONFIG['history_stride_s'], help='seconds')
    parser.add_argument('--steps', type=float, default=default_steps)
    parser.add_argument('--n-envs', type=int, default=default_envs)
    parser.add_argument('--n-steps', type=int, default=SETTINGS['n_steps'], help='rollout per env')
    parser.add_argument('--n-epochs', type=int, default=SETTINGS['n_epochs'])
    parser.add_argument('--batch-size', default=SETTINGS['batch_size'],
                        help="an int, or 'full' for the whole rollout buffer")
    parser.add_argument('--net-width', type=int, default=128, help='neurons per layer of the 2-layer heads')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', default=None)
    parser.add_argument('--resume', default=None, help='a .zip; its vecnormalize.pkl is read beside it')
    return parser.parse_args()


def run_config(args):
    return {**CONFIG, 'action_frequency': args.decision_step, 'history_stride_s': args.history_stride}


def load_cfg(model_path):
    """The cfg a saved model was trained with, from config.json in its run folder; the
    checkpoints/ folder sits one level deeper. CONFIG for the neutral policy."""
    if model_path is None:
        return CONFIG
    for folder in Path(model_path).parents[:2]:
        if (folder / 'config.json').exists():
            return json.loads((folder / 'config.json').read_text())['cfg']
    raise FileNotFoundError(f'no config.json beside {model_path}')


def train(kind, args, cfg, venv, policy, policy_kwargs, evaluate):
    """Train one run: kind is 'sa' or 'ma', venv the un-normalised VecEnv, evaluate(model, n)
    returns a dict of KPIs over n validation episodes."""
    torch.set_num_threads(2)
    full = args.batch_size == 'full'
    settings = {**SETTINGS, 'n_steps': args.n_steps, 'n_epochs': args.n_epochs,
                'batch_size': args.n_steps * venv.num_envs if full else int(args.batch_size)}
    run_dir = Path('Models') / kind / (args.name or f'{args.env}_seed{args.seed}')
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.json').write_text(json.dumps(
        {'cfg': cfg, 'settings': settings, 'n_envs': args.n_envs, 'env': args.env}, indent=2))

    if args.resume:
        venv = VecNormalize.load(str(Path(args.resume).parent / 'vecnormalize.pkl'), venv)
        model = PPO.load(args.resume, venv, tensorboard_log=str(run_dir))
    else:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=settings['gamma'])
        model = PPO(policy, venv, policy_kwargs=policy_kwargs, tensorboard_log=str(run_dir),
                    seed=args.seed, verbose=1, **settings)

    callbacks = [CheckpointCallback(max(1, SAVE_EVERY // venv.num_envs), str(run_dir / 'checkpoints'),
                                    save_vecnormalize=True),
                 TrainingKPIs(),
                 EvaluationKPIs(evaluate, run_dir)]
    model.learn(int(args.steps), callback=callbacks, reset_num_timesteps=not args.resume)
    model.save(run_dir / 'final_model')
    venv.save(str(run_dir / 'vecnormalize.pkl'))
    venv.close()


def kpi_means(records):
    kpis = {key: float(np.mean([r[key] for r in records])) for key in METRIC_KEYS}
    kpis['goal_completion_rate_pct'] = 100.0 * kpis['waypoint_reached']
    return kpis


class TrainingKPIs(BaseCallback):
    """The KPIs of every training episode as it ends, averaged per rollout."""

    def _on_step(self):
        for info in self.locals['infos']:
            if 'episode' in info and 'waypoint_reached' in info:
                for key, value in kpi_means([info]).items():
                    self.logger.record_mean(f'rollout/{key}', value)
        return True


class EvaluationKPIs(BaseCallback):
    """Deterministic episodes on held-out scenarios every EVAL_EVERY steps."""

    def __init__(self, evaluate, run_dir):
        super().__init__()
        self.evaluate, self.run_dir = evaluate, run_dir
        self.next_eval, self.best = EVAL_EVERY, -np.inf

    def _on_step(self):
        if self.num_timesteps < self.next_eval:
            return True
        self.next_eval += EVAL_EVERY
        kpis = self.evaluate(self.model, EVAL_EPISODES)
        for key, value in kpis.items():
            self.logger.record(f'eval/{key}', value)
        if kpis['total_reward'] > self.best:
            self.best = kpis['total_reward']
            self.model.save(self.run_dir / 'best_model')
        return True
