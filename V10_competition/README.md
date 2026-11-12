# V10 competition agents

V2's adjusted PPO setup with the V8 reward. Both tracks use the V2 lidar,
5-second action interval, and per-aircraft attention critic. The actor uses
its own observation. BlueSky is the default for training and periodic
validation. Use `--env 2d` to select the 2D surrogate.

| Setting | Single-agent | Multi-agent |
| --- | ---: | ---: |
| Parallel scenarios | 8 | 4 |
| Aircraft slots per scenario | 1 | 10 |
| Decisions per slot per rollout | 100 | 100 |
| Full batch size | 800 | 4,000 |
| Epochs | 30 | 30 |

Learning rate is 0.0003, discount gamma is 0.99, and GAE lambda is 0.95.
The training wrapper scales rewards using running return variance.

The reward uses increments in competition metrics. Each aircraft pays
0.5 per second in loss of separation, 3 per new intrusion, 0.2 per second
in restricted airspace, 2 per restricted-area entry, 0.3 per second
outside the sector, and 0.015 per flight second. In MA, each aircraft is
charged the team's summed intrusion metrics and its own airspace and
flight costs. The V8 formula has no heading drift term.

## Train

From this folder, use an existing environment with the project dependencies,
or create one with `uv sync`. Then run:

```bash
uv run python -u -m single_agent.Training.train --name ppo_adjusted
uv run python -u -m multi_agent.Training.train --name mappo_adjusted
```

Training runs until stopped. Pass `--steps` to set a finite limit. Press
Ctrl+C to stop and save `final_model.zip` and `vecnormalize.pkl`. Model
and reward-normalization checkpoints are saved every 20,000 aircraft steps
while training continues. Use `--save-every` to change that interval.
Checkpoints, validation-selected models, and TensorBoard logs go under
`Models/sa/` and `Models/ma/`.
