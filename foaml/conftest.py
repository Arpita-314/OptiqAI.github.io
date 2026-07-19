"""Make the flat intra-package imports (``models.*``, ``optics.*``, ...) resolve
by putting this directory on ``sys.path`` for the whole test session.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
