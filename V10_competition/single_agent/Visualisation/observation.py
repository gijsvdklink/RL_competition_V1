"""Plot what the single-agent policy observes, one figure per decision step.

Each figure decodes the observation dict the env returns (Shared/observation.py):

    scene          top-down view: sector, restricted areas, intruders, goal, and the agent's
                   newest lidar scan drawn as rays to their hits, coloured by channel; the
                   agent positions of the 4 stacked snapshots are marked
    lidar (polar)  the newest scan around the nose, one colour per channel
    state          the 9 state values of each of the 4 snapshots, oldest light, newest dark
    lidar history  the 3 lidar channels of all 4 snapshots, rays against time
    critic tokens  the 'joint' input of the centralised critic, one row per aircraft;
                   training only, the actor never sees it

    python -m single_agent.Visualisation.observation --seed 1 --steps 0 20 40 60
    python -m single_agent.Visualisation.observation --model Models/sa/ppo_default/best_model.zip --seed 1 --steps 0 50 100
    python -m single_agent.Visualisation.observation --seed 1 --gif --every 2      # whole episode as a GIF

Without --model the agent flies straight (the neutral policy). Runs on the 2D surrogate.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

from Shared.geo import to_local_nm
from Shared.observation import JOINT_DIM, N_CHANNELS, STATE_DIM
from Shared.training import load_cfg
from single_agent.Training.train import make_env
from single_agent.Validation.evaluate import load_policy

STATE_NAMES = ['progress', 'cos_angle', 'speed_ratio', 'closing', 'distance',
               'goal_cos', 'goal_sin', 'cmd_heading', 'cmd_speed']
JOINT_NAMES = ['alive', 'self'] + STATE_NAMES + ['rel_N', 'rel_E', 'rel_vN', 'rel_vE']
CHANNELS = [('restricted area / boundary', '#d62728', 'Reds'),
            ('aircraft', '#1f77b4', 'Blues'),
            ('destination', '#2ca02c', 'Greens')]
N_HITS = 3          # channels 0-2 are range hits; channel 3 is the goal bump (Shared/lidar.py)
AGENT_COLOUR = '#ff7f0e'
GOAL_COLOUR = '#2ca02c'

assert len(JOINT_NAMES) == JOINT_DIM and N_CHANNELS in (N_HITS, N_HITS + 1)
CHANNELS = CHANNELS[:N_CHANNELS]     # this version's lidar has no goal channel


def decode(own, cfg):
    """The actor's flat observation as (state (H, 9), lidar (H, 3, n_rays)), oldest first."""
    frames = own.reshape(cfg['history_length'], -1)
    lidar = frames[:, STATE_DIM:].reshape(cfg['history_length'], N_CHANNELS, cfg['n_rays'])
    return frames[:, :STATE_DIM], lidar


def ray_offsets(cfg):
    """The ray angles relative to the nose in degrees, as Shared/lidar.py spaces them."""
    return np.linspace(-cfg['fov_deg'] / 2, cfg['fov_deg'] / 2, cfg['n_rays'],
                       endpoint=cfg['fov_deg'] < 360.0)


def snapshot_labels(cfg):
    h, stride = cfg['history_length'], cfg['history_stride_s']
    return [f't-{(h - 1 - k) * stride} s' if k < h - 1 else 't' for k in range(h)]


def local(points_latlon, center):
    """(lat, lon) rows to (east, north) NM rows, for plotting with east on x."""
    pts = np.asarray(points_latlon, dtype=float)
    return to_local_nm(pts[:, 0], pts[:, 1], center)[:, ::-1]


def draw_scene(ax, env, cfg, lidar_now, agent_trail, snapshot_steps, step):
    scenario, traffic = env.scenario, env.traffic
    center = scenario.center

    sector = local(scenario.sector, center)
    ax.add_patch(Polygon(sector, closed=True, fill=False, ec=CHANNELS[1][1], lw=1.5))
    for obstacle in scenario.obstacles:
        ax.add_patch(Polygon(local(obstacle.vertices, center), closed=True,
                             fc=CHANNELS[0][1], ec=CHANNELS[0][1], alpha=0.35))

    goal = local([scenario.agents[0].goal], center)[0]
    ax.plot(*goal, marker='*', ms=16, color=GOAL_COLOUR, mec='k', zorder=5, label='goal')

    pos = to_local_nm(traffic.lat, traffic.lon, center)[:, ::-1]
    i_agent = traffic.id2idx(env.agent)
    for j, ac in enumerate(traffic.id):
        if j == i_agent:
            continue
        ax.add_patch(Circle(pos[j], cfg['aircraft_radius_nm'], fc=CHANNELS[2][1], alpha=0.25,
                            ec=CHANNELS[2][1]))
        _arrow(ax, pos[j], traffic.hdg[j], 4.0, CHANNELS[2][1])
        ax.annotate(ac, pos[j], xytext=(4, 4), textcoords='offset points', fontsize=7,
                    color=CHANNELS[2][1])

    # rays of the newest scan, to their hit or to full range
    origin, hdg = pos[i_agent], traffic.hdg[i_agent]
    angles = np.radians(hdg + ray_offsets(cfg))
    direction = np.stack([np.sin(angles), np.cos(angles)], axis=1)
    strength = lidar_now[:N_HITS].max(axis=0)
    channel = lidar_now[:N_HITS].argmax(axis=0)
    rng = cfg['lidar_range_nm']
    for k in range(cfg['n_rays']):
        if strength[k] > 0:
            end = origin + direction[k] * rng * (1.0 - strength[k])
            colour = CHANNELS[channel[k]][1]
            ax.plot(*np.stack([origin, end]).T, color=colour, lw=0.9, alpha=0.8)
            ax.plot(*end, 'o', ms=2.5, color=colour)
        else:
            end = origin + direction[k] * rng
            ax.plot(*np.stack([origin, end]).T, color='#bbbbbb', lw=0.5, alpha=0.5)
    ax.add_patch(Circle(origin, rng, fill=False, ls=':', ec='#999999', lw=0.8))

    # where the agent was at each stacked snapshot
    ax.plot(*np.array(agent_trail).T, color=AGENT_COLOUR, lw=1, alpha=0.5)
    shades = np.linspace(0.3, 1.0, len(snapshot_steps))
    for s, label, alpha in zip(snapshot_steps, snapshot_labels(cfg), shades):
        ax.plot(*agent_trail[s], 'o', ms=7, color=AGENT_COLOUR, alpha=alpha, mec='k', mew=0.5)
        if label != 't':
            ax.annotate(label, agent_trail[s], xytext=(-6, -12), textcoords='offset points',
                        fontsize=7, color=AGENT_COLOUR)
    ax.add_patch(Circle(origin, cfg['aircraft_radius_nm'], fc=AGENT_COLOUR, ec='k', zorder=6))
    _arrow(ax, origin, hdg, 6.0, 'k')
    _arrow(ax, origin, traffic.cmd_hdg[i_agent], 6.0, AGENT_COLOUR, ls='--')

    lo, hi = sector.min(axis=0), sector.max(axis=0)
    pad = 0.05 * (hi - lo).max()
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect('equal')
    ax.set_xlabel('east [NM]')
    ax.set_ylabel('north [NM]')
    handles = [plt.Line2D([], [], color=c, lw=2, label=n) for n, c, _ in CHANNELS[:N_HITS]]
    handles += [plt.Line2D([], [], color=AGENT_COLOUR, marker='o', lw=0, label='agent (snapshots)'),
                plt.Line2D([], [], color=AGENT_COLOUR, ls='--', label='commanded heading'),
                plt.Line2D([], [], color=GOAL_COLOUR, marker='*', lw=0, ms=12, label='goal')]
    ax.legend(handles=handles, loc='upper right', fontsize=7, framealpha=0.85)
    ax.set_title(f'Scene and newest lidar scan (step {step}, '
                 f't = {step * cfg["action_frequency"]} s)')


def _arrow(ax, xy, hdg_deg, length, colour, ls='-'):
    h = np.radians(hdg_deg)
    end = xy + length * np.array([np.sin(h), np.cos(h)])
    ax.annotate('', end, xy, arrowprops=dict(arrowstyle='->', color=colour, lw=1.4, ls=ls))


def draw_polar(ax, lidar_now, cfg):
    theta = np.radians(ray_offsets(cfg))
    width = 2 * np.pi / cfg['n_rays'] if cfg['fov_deg'] >= 360 else np.radians(cfg['fov_deg']) / cfg['n_rays']
    for c, (name, colour, _) in enumerate(CHANNELS[:N_HITS]):
        ax.bar(theta, lidar_now[c], width=width, color=colour, alpha=0.8, label=name)
    if N_CHANNELS > N_HITS:
        name, colour, _ = CHANNELS[N_HITS]
        ax.plot(np.append(theta, theta[0]), np.append(lidar_now[N_HITS], lidar_now[N_HITS][0]),
                color=colour, lw=2, label=name)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.tick_params(labelsize=7)
    ax.set_title('Newest lidar, nose up\n(1 - hit distance / range)', fontsize=10)


def draw_state(ax, state, cfg):
    h = len(state)
    y = np.arange(STATE_DIM)
    height = 0.8 / h
    shades = np.linspace(0.3, 1.0, h)
    for k, (label, alpha) in enumerate(zip(snapshot_labels(cfg), shades)):
        ax.barh(y - 0.4 + (k + 0.5) * height, state[k], height=height, color=AGENT_COLOUR,
                alpha=alpha, label=label)
    for yi, v in zip(y, state[-1]):
        ax.annotate(f'{v:+.2f}', (v, yi + 0.4 - 0.5 * height), xytext=(3 if v >= 0 else -3, 0),
                    textcoords='offset points', fontsize=7, va='center',
                    ha='left' if v >= 0 else 'right')
    ax.axvline(0, color='k', lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(STATE_NAMES, fontsize=8)
    ax.invert_yaxis()
    ax.legend(fontsize=7, loc='lower right')
    ax.set_title('State vector, 4 snapshots', fontsize=10)


def draw_lidar_history(axes, lidar, cfg):
    offsets = ray_offsets(cfg)
    step = offsets[1] - offsets[0]
    extent = [offsets[0] - step / 2, offsets[-1] + step / 2, len(lidar) - 0.5, -0.5]
    for c, (ax, (name, _, cmap)) in enumerate(zip(axes, CHANNELS)):
        im = ax.imshow(lidar[:, c, :], aspect='auto', cmap=cmap, vmin=0, vmax=1, extent=extent,
                       interpolation='nearest')
        ax.set_yticks(range(len(lidar)))
        ax.set_yticklabels(snapshot_labels(cfg), fontsize=8)
        ax.set_xlabel('ray angle from nose [deg]')
        ax.set_xticks(np.arange(-180, 181, 90))
        ax.set_title(f'Lidar history: {name}', fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)


def draw_joint(ax, joint, slots):
    im = ax.imshow(joint, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    for (r, c), v in np.ndenumerate(joint):
        if v != 0:
            ax.text(c, r, f'{v:.2f}', ha='center', va='center', fontsize=5.5,
                    color='w' if abs(v) > 0.6 else 'k')
    ax.set_xticks(range(JOINT_DIM))
    ax.set_xticklabels(JOINT_NAMES, rotation=60, ha='right', fontsize=7)
    ax.set_yticks(range(len(slots)))
    ax.set_yticklabels(slots, fontsize=7)
    ax.set_title("Critic tokens ('joint', training only): agent first, then intruders",
                 fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)


def figure(env, cfg, obs, action, agent_trail, step, seed):
    state, lidar = decode(obs['own'], cfg)
    stride = cfg['history_stride_s'] // cfg['action_frequency']
    snapshot_steps = [max(0, step - k * stride) for k in reversed(range(cfg['history_length']))]

    fig = plt.figure(figsize=(20, 15))
    grid = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.55], hspace=0.45, wspace=0.35)
    draw_scene(fig.add_subplot(grid[0:2, 0:2]), env, cfg, lidar[-1], agent_trail,
               snapshot_steps, step)
    draw_polar(fig.add_subplot(grid[0, 2], projection='polar'), lidar[-1], cfg)
    draw_state(fig.add_subplot(grid[0, 3]), state, cfg)
    draw_joint(fig.add_subplot(grid[1, 2:4]), obs['joint'], [env.agent] + env.intruder_ids)
    sub = grid[2, :].subgridspec(1, N_CHANNELS, wspace=0.3)
    draw_lidar_history([fig.add_subplot(sub[0, c]) for c in range(N_CHANNELS)], lidar, cfg)
    fig.suptitle(f'Single-agent observation: seed {seed}, step {step}; '
                 f'actor input {obs["own"].size} values = {cfg["history_length"]} x '
                 f'({STATE_DIM} state + {N_CHANNELS} x {cfg["n_rays"]} lidar); '
                 f'action taken [hdg {action[0]:+.2f}, spd {action[1]:+.2f}]', fontsize=13)
    return fig


def to_image(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--steps', type=int, nargs='+', default=[0, 20, 40, 60],
                        help='decision steps to save as PNG')
    parser.add_argument('--gif', action='store_true', help='also save the whole episode as a GIF')
    parser.add_argument('--mp4', action='store_true', help='also save the whole episode as an MP4')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--every', type=int, default=2, help='GIF/MP4: keep every n-th step')
    parser.add_argument('--out', default='Results/sa/observations')
    args = parser.parse_args()

    cfg = load_cfg(args.model)
    act = load_policy(args.model)
    env = make_env('2d', cfg)
    run = Path(args.model).parent.name if args.model else 'neutral'
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    movie = args.gif or args.mp4
    obs, _ = env.reset(seed=args.seed)
    wanted, frames, trail = set(args.steps), [], []
    step, done, truncated = 0, False, False
    while not (done or truncated):
        i = env.traffic.id2idx(env.agent)
        trail.append(to_local_nm(env.traffic.lat[i], env.traffic.lon[i], env.scenario.center)[::-1])
        action = np.asarray(act(obs), dtype=float)
        if step in wanted or (movie and step % args.every == 0):
            fig = figure(env, cfg, obs, action, trail, step, args.seed)
            if step in wanted:
                path = out / f'{run}_seed{args.seed}_step{step:04d}.png'
                fig.savefig(path, dpi=110)
                print(f'wrote {path}')
            if movie and step % args.every == 0:
                frames.append(to_image(fig))
            plt.close(fig)
        obs, _, done, truncated, info = env.step(action)
        step += 1
        if not movie and step > max(wanted):
            break

    missed = sorted(s for s in wanted if s >= step)
    if missed:
        print(f'episode ended at step {step}; no figure for steps {missed}')
    if args.gif and frames:
        from PIL import Image
        images = [Image.fromarray(f).resize((f.shape[1] // 2, f.shape[0] // 2)) for f in frames]
        path = out / f'{run}_seed{args.seed}.gif'
        images[0].save(path, save_all=True, append_images=images[1:], duration=150, loop=0)
        print(f'wrote {path} ({len(images)} frames)')
    if args.mp4 and frames:
        from Shared.video import save_mp4
        save_mp4(frames, out / f'{run}_seed{args.seed}.mp4', fps=args.fps)


if __name__ == '__main__':
    main()
