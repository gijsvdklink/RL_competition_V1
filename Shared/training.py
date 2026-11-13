"""What both trainers share: the PPO settings, the command line, and the TensorBoard KPIs.

Both train with Stable-Baselines3 PPO in the competition's own BlueSky environments.
Training episodes end on a new violation (CONFIG terminate_on_violation); the evaluation
episodes and the official harness keep the competition's rules. The update schedule:

    one environment       in its own process: the evaluation environment lives in the main
                          one, and BlueSky can only run once per process
    rollout of 1000       samples: 1000 decisions of the single agent, or 100 decisions of
                          each of the 10 aircraft in the multi-agent env
    batch = the rollout   the whole rollout buffer is one batch
    30 epochs             stopped early once the policy has moved by target_kl 0.02

The settings follow navigation-mappo-rl where they carry over: learning rate 5e-4, clip 0.2,
GAE lambda 0.95, no entropy bonus. gamma is 0.98 per 10 s decision, a horizon of about
8 minutes: the repo's 0.9 would be 100 s, too short for a conflict that has to be resolved
minutes ahead.

Normalisation is SB3's own: VecNormalize scales observations and rewards by running
statistics, and PPO normalises the advantages per batch. The observation statistics are part
of the model: they are saved beside every model as <model>_vecnormalize.pkl (for checkpoints
rl_model_vecnormalize_<N>_steps.pkl), and load_policy applies them before the policy acts.

TensorBoard, under Models/<sa|ma>/<run>/:

    rollout/ep_rew_mean  the total reward per (agent-)episode, SB3's own, over the last 100
    rollout/<kpi>        the goal completion rate, the nine scored metrics and the share of
                         episodes ended by a violation: the mean over the episodes that ended
                         in that rollout, with rollout/<kpi>_std their standard deviation
    eval/*               every EVAL_EVERY steps, EVAL_EPISODES deterministic episodes on
                         held-out scenarios (seed 7, never the competition's 42), under the
                         competition's rules: no termination on a violation

The model with the best eval/total_reward is kept as best_model.zip.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

from Shared.config import CONFIG

SETTINGS = {
    'learning_rate':        5e-4,
    'n_epochs':             30,
    'target_kl':            0.02,
    'gamma':                0.98,
    'gae_lambda':           0.95,
    'clip_range':           0.2,
    'ent_coef':             0.0,
    'vf_coef':              0.5,
    'max_grad_norm':        0.5,
}

ROLLOUT_SAMPLES = 1000             # the rollout buffer, and the batch
LOG_STD_INIT    = -0.7             # action spread 0.5: still flies to the goal, as the baseline
SAVE_EVERY      = 500_000          # steps
EVAL_EVERY      = 100_000          # steps
EVAL_EPISODES   = 20
VALIDATION_SEED = 7                # the eval/* scenarios; the competition scores on seed 42
TRAINING_SEEDS  = 10_000           # run --seed s trains on seeds 10_000 (s + 1) + env index

METRIC_KEYS = ('total_reward', 'flight_time', 'waypoint_reached',
               'intrusion_events', 'intrusion_time',
               'sector_exit_events', 'time_outside_sector',
               'restricted_area_events', 'time_in_restricted_area')


def parse_args(doc, default_steps):
    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--steps', type=float, default=default_steps)
    parser.add_argument('--net-width', type=int, default=128, help='neurons per layer of the 2-layer heads')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', default=None)
    parser.add_argument('--resume', default=None, help='a final, best or checkpoint .zip')
    return parser.parse_args()


def evaluation_cfg(cfg):
    """The competition's rules: no termination on a violation."""
    return {**cfg, 'terminate_on_violation': False}


def load_cfg(model_path):
    """The cfg a saved model was trained with, from config.json in its run folder; the
    checkpoints/ folder sits one level deeper. CONFIG for the neutral policy."""
    if model_path is None:
        return CONFIG
    for folder in Path(model_path).parents[:2]:
        if (folder / 'config.json').exists():
            return {**CONFIG, **json.loads((folder / 'config.json').read_text())['cfg']}
    raise FileNotFoundError(f'no config.json beside {model_path}')


def train(kind, args, cfg, venv, policy, policy_kwargs, evaluate):
    """Train one run: kind is 'sa' or 'ma', evaluate(model, n) returns a dict of KPIs over n
    validation episodes."""
    torch.set_num_threads(2)
    settings = {**SETTINGS, 'n_steps': ROLLOUT_SAMPLES // venv.num_envs, 'batch_size': ROLLOUT_SAMPLES}
    run_dir = Path('Models') / kind / (args.name or f'seed{args.seed}')
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        venv = VecNormalize.load(str(stats_path(args.resume)), venv)
        model = PPO.load(args.resume, venv, tensorboard_log=str(run_dir))
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, gamma=settings['gamma'])
        model = PPO(policy, venv, policy_kwargs={**policy_kwargs, 'log_std_init': LOG_STD_INIT},
                    tensorboard_log=str(run_dir), seed=args.seed, verbose=1, **settings)
    # PPO seeds env i with --seed + i, which repeats across runs; give every env its own
    # stream instead, moved on by the steps already trained so a resumed run sees new scenarios
    venv.seed(TRAINING_SEEDS * (args.seed + 1) + model.num_timesteps)
    used = {key: getattr(model, key) for key in ('n_steps', 'batch_size', 'n_epochs')}
    (run_dir / 'config.json').write_text(json.dumps(
        {'cfg': cfg, 'settings': {**settings, **used}}, indent=2))

    callbacks = [CheckpointCallback(max(1, SAVE_EVERY // venv.num_envs), str(run_dir / 'checkpoints'),
                                    save_vecnormalize=True),
                 TrainingKPIs(),
                 EvaluationKPIs(evaluate, run_dir)]
    model.learn(int(args.steps), callback=callbacks, reset_num_timesteps=not args.resume)
    save(model, run_dir / 'final_model')
    venv.close()


def save(model, path):
    """The model and the observation statistics it needs."""
    model.save(path)
    model.get_vec_normalize_env().save(str(stats_path(path)))


def stats_path(model_path):
    """The VecNormalize statistics saved beside a model."""
    path = Path(model_path)
    if path.name.startswith('rl_model_'):
        return path.with_name(path.name.replace('rl_model_', 'rl_model_vecnormalize_')).with_suffix('.pkl')
    return path.with_name(path.stem.removesuffix('.zip') + '_vecnormalize.pkl')


def load_policy(model_path):
    """act(obs) -> action for one aircraft: the observation is normalised with the model's
    statistics, then the deterministic action is taken. The fly-straight [0, 0] without a model."""
    if model_path is None:
        return lambda obs: np.zeros(2)
    model = PPO.load(model_path, device='cpu')
    with open(stats_path(model_path), 'rb') as f:
        stats = pickle.load(f)
    stats.training, stats.norm_reward = False, False
    return lambda obs: model.predict(stats.normalize_obs(obs), deterministic=True)[0]


def kpi_means(records):
    kpis = {key: float(np.mean([r[key] for r in records])) for key in METRIC_KEYS}
    kpis['goal_completion_rate_pct'] = 100.0 * kpis['waypoint_reached']
    if all('ended_by_violation' in r for r in records):
        kpis['ended_by_violation_pct'] = 100.0 * float(np.mean([r['ended_by_violation'] for r in records]))
    return kpis


class TrainingKPIs(BaseCallback):
    """The KPIs of the training episodes that ended in a rollout: rollout/<kpi> is their
    mean and rollout/<kpi>_std their standard deviation, as the competition's table reports."""

    def __init__(self):
        super().__init__()
        self.episodes = []

    def _on_step(self):
        self.episodes += [info for info in self.locals['infos']
                          if 'episode' in info and 'waypoint_reached' in info]
        return True

    def _on_rollout_end(self):
        if not self.episodes:
            return
        per_episode = [kpi_means([info]) for info in self.episodes]
        for key in per_episode[0]:
            values = [kpis[key] for kpis in per_episode]
            self.logger.record(f'rollout/{key}', float(np.mean(values)))
            self.logger.record(f'rollout/{key}_std', float(np.std(values)))
        self.logger.record('rollout/episodes', len(per_episode))
        self.episodes = []


class EvaluationKPIs(BaseCallback):
    """Deterministic episodes on held-out scenarios every EVAL_EVERY steps."""

    def __init__(self, evaluate, run_dir):
        super().__init__()
        self.evaluate, self.run_dir = evaluate, run_dir
        self.best = -np.inf

    def _on_training_start(self):
        self.next_eval = (self.model.num_timesteps // EVAL_EVERY + 1) * EVAL_EVERY

    def _on_step(self):
        if self.num_timesteps < self.next_eval:
            return True
        self.next_eval += EVAL_EVERY
        kpis = self.evaluate(self.model, EVAL_EPISODES)
        for key, value in kpis.items():
            self.logger.record(f'eval/{key}', value)
        if kpis['total_reward'] > self.best:
            self.best = kpis['total_reward']
            save(self.model, self.run_dir / 'best_model')
        return True
