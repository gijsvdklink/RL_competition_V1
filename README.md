# RL agents for the BlueSky-Gym AI4REAL-NET competition

This repository holds PPO (single-agent) and MAPPO (multi-agent) policies for the
[BlueSky-Gym competition](https://github.com/TUDelft-CNS-ATM/bluesky-gym/blob/AI4REAL-NET-Competition/docs/competition/COMPETITION.md).
The observation and network follow [navigation-mappo-rl](https://github.com/avbiswas/navigation-mappo-rl):
- a lidar with channels for restricted areas, the sector boundary and other aircraft, plus a
  fourth channel that marks the direction of the goal;
- a state vector, extended with the commanded heading and the commanded speed;
- 4 stacked snapshots.

Both use the same network (`Shared/policy.py`), the MAPPO design of the repo:
- a decentralised actor that flies on the aircraft's own observation;
- a centralised critic that also attends over the true states of all other aircraft. In the
  single-agent env those are the 10 intruders; in the multi-agent env, the other agents.

Training runs on a fast 2D surrogate of BlueSky. It reproduces the competition's scenarios,
actions and scoring, and has been validated against BlueSky. Evaluation runs on the official
harness in BlueSky itself.

## Layout

| folder | contents |
|---|---|
| `Shared/` | config, surrogate traffic model, lidar, observation, reward, scoring copy, encoder, policy, SB3 training loop |
| `single_agent/` | surrogate and BlueSky environments, PPO training, evaluation, GIF recording |
| `multi_agent/` | the same for 10 agents, with one policy shared by all aircraft |
| `Validation/` | `calibrate_dynamics.py` (measures BlueSky's A320) and `validate_surrogate.py` (surrogate vs BlueSky) |
| `Docs/` | the reward function |

## Install

The competition code must sit next to this repository, on its competition branch:

```bash
git clone --branch AI4REAL-NET-Competition https://github.com/TUDelft-CNS-ATM/bluesky-gym.git
git clone https://github.com/gijsvdklink/RL_competition_V1.git competition-rl
cd competition-rl
UV_PROJECT_ENVIRONMENT=~/.venvs/competition-rl uv sync
source ~/.venvs/competition-rl/bin/activate
```

The venv is kept outside the project on purpose. macOS hides `.pth` files inside an
iCloud-synced Documents folder, and the editable install of bluesky-gym then fails to import.
torch is pinned to 2.2.2, the last release with Intel-Mac wheels.

## Train

Run everything from this folder. One update schedule, as in the thesis:
- one environment;
- a decision every 5 s;
- a rollout buffer of 1000 samples: 1000 decisions for PPO, 100 decisions of each of the 10
  aircraft for MAPPO;
- the whole rollout as one batch;
- 30 epochs.

```bash
python -m single_agent.Training.train --env 2d --steps 1e6 --name ppo
python -m multi_agent.Training.train  --env 2d --steps 5e6 --name mappo
python -m tensorboard.main --logdir Models --port 6006
```

TensorBoard shows three groups:
- **`rollout/*`:** the reward and the competition KPIs of the training episodes;
- **`eval/*`:** deterministic episodes on held-out scenarios, every 100k steps;
- **`train/*`:** PPO's own diagnostics.

## Evaluate and record

```bash
python -m single_agent.Validation.evaluate --model Models/sa/ppo/best_model.zip      # official harness, BlueSky
python -m single_agent.Validation.evaluate --model Models/sa/ppo/best_model.zip --env 2d --episodes 100
python -m single_agent.Visualisation.visualise --model Models/sa/ppo/best_model.zip --seeds 1 2 3 4 5
```

For the multi-agent model, use the `multi_agent.` modules and a model under `Models/ma/`.
