"""Access to the competition's own scripts/, copied into this folder from bluesky-gym.

The evaluation harness and the GIF writer are imported from there unchanged, so reported
numbers and videos come from the official code.
"""

import importlib
import sys
from pathlib import Path

BLUESKY_GYM_DIR = Path(__file__).resolve().parents[1]


def competition_script(name):
    if str(BLUESKY_GYM_DIR) not in sys.path:
        sys.path.insert(0, str(BLUESKY_GYM_DIR))
    return importlib.import_module(f'scripts.{name}')
