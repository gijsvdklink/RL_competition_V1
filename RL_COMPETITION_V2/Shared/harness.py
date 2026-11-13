"""Access to the competition's own scripts/ in ../bluesky-gym, which pip does not install.

The evaluation harness and the GIF writer are imported from there unchanged, so reported
numbers and videos come from the official code.
"""

import importlib
import sys
from pathlib import Path

BLUESKY_GYM_DIR = Path(__file__).resolve().parents[2] / 'bluesky-gym'


def competition_script(name):
    if str(BLUESKY_GYM_DIR) not in sys.path:
        sys.path.insert(0, str(BLUESKY_GYM_DIR))
    return importlib.import_module(f'scripts.{name}')
