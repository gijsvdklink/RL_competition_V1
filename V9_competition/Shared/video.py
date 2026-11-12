"""Write rendered frames as an MP4, with every decision step kept (the GIF writer subsamples).

Needs imageio and imageio-ffmpeg, which bundles its own ffmpeg:

    pip install imageio imageio-ffmpeg
"""

from pathlib import Path


def save_mp4(frames, out_path, fps=20):
    """frames: (H, W, 3) uint8 arrays. At the default 20 fps and a decision every 5 s, one
    second of video is 100 s of flight."""
    import imageio.v2 as imageio

    if not frames:
        raise ValueError('no frames to save')
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out_path, fps=fps, codec='libx264', quality=8,
                            pixelformat='yuv420p', macro_block_size=16) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f'wrote {out_path}  ({len(frames)} frames, {len(frames) / fps:.1f} s)')
