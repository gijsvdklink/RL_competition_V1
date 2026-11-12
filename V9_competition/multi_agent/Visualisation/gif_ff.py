"""One multi-agent episode as a GIF with a fast-forward overlay.

Each frame carries a ▶▶ badge with the speed-up over real time, the simulated clock, and how
many aircraft are still flying and how many have reached their goal. Frames are those of
visualise.py (Shared/render.py).

    python -m multi_agent.Visualisation.gif_ff --model Models/ma/mappo_adjusted/best_model.zip --seed 1

--every keeps every n-th decision step (5 s each) and --ms is the time per GIF frame, so the
speed-up is 5 * every * 1000 / ms (default 3 steps per 60 ms: 250x).
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from Shared.render import Renderer
from Shared.training import load_cfg
from multi_agent.Training.train import make_env
from multi_agent.Validation.evaluate import load_policy


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                       # Pillow < 10.1
        return ImageFont.load_default()


def overlay(frame, speedup, t, flying, reached, n_agents, done):
    img = Image.fromarray(frame).convert('RGBA')
    layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    w = img.size[0]

    # ▶▶ badge, top right
    x0, y0, bw, bh = w - 150, 10, 140, 40
    d.rounded_rectangle([x0, y0, x0 + bw, y0 + bh], radius=10, fill=(0, 0, 0, 170))
    for k in range(2):
        tx = x0 + 12 + k * 16
        d.polygon([(tx, y0 + 10), (tx, y0 + bh - 10), (tx + 16, y0 + bh / 2)],
                  fill=(255, 255, 255, 255) if not done else (120, 220, 120, 255))
    d.text((x0 + 52, y0 + bh / 2), 'END' if done else f'{speedup:.0f}x', font=font(20),
           fill=(255, 255, 255, 255), anchor='lm')

    # clock and progress bar, top left
    d.rounded_rectangle([10, 10, 250, 64], radius=10, fill=(0, 0, 0, 170))
    d.text((22, 24), f't = {int(t) // 60:02d}:{int(t) % 60:02d}', font=font(20),
           fill=(255, 255, 255, 255), anchor='lm')
    d.text((22, 50), f'flying {flying}/{n_agents}   goal {reached}/{n_agents}', font=font(14),
           fill=(220, 220, 220, 255), anchor='lm')
    bar = int(222 * min(t / 3000.0, 1.0))
    d.rectangle([18, 60, 18 + bar, 62], fill=(255, 140, 0, 255))
    return Image.alpha_composite(img, layer).convert('RGB')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--every', type=int, default=3, help='keep every n-th decision step')
    parser.add_argument('--ms', type=int, default=60, help='milliseconds per GIF frame')
    parser.add_argument('--size', type=int, default=640)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    cfg = load_cfg(args.model)
    act = load_policy(args.model)
    env = make_env('2d', cfg)
    speedup = cfg['action_frequency'] * args.every * 1000 / args.ms

    observations, _ = env.reset(seed=args.seed)
    renderer = Renderer(env.scenario, cfg, size=args.size)
    goals = {spec.ac_id: tuple(spec.goal) for spec in env.scenario.agents}
    n_agents, step, records, images = len(env.agents), 0, [], []

    def snap(done=False):
        frame = renderer.frame(env.traffic, list(env.agents), goals,
                               {a: observations[a]['own'] for a in env.agents})
        reached = int(sum(r['waypoint_reached'] for r in records))
        images.append(overlay(frame, speedup, step * cfg['action_frequency'], len(env.agents),
                              reached, n_agents, done))

    while env.agents:
        if step % args.every == 0:
            snap()
        observations, _, terms, truncs, infos = env.step({a: act(observations[a]) for a in env.agents})
        records += [infos[a] for a in infos if terms[a] or truncs[a]]
        step += 1
    snap(done=True)

    keys = ('waypoint_reached', 'intrusion_time', 'time_in_restricted_area', 'flight_time')
    print(f'seed {args.seed}: ' + ', '.join(f'{k} {np.mean([r[k] for r in records]):.2f}' for k in keys))

    run = Path(args.model).parent.name if args.model else 'neutral'
    out = Path(args.out or f'Results/ma/{run}_seed{args.seed}_ff.gif')
    out.parent.mkdir(parents=True, exist_ok=True)
    durations = [args.ms] * (len(images) - 1) + [2000]          # hold the last frame
    images[0].save(out, save_all=True, append_images=images[1:], duration=durations, loop=0,
                   optimize=True)
    print(f'wrote {out}  ({len(images)} frames, {speedup:.0f}x)')


if __name__ == '__main__':
    main()
