"""
STWR: Spatiotemporal Weighted Regression

A Python package for calibrating geographically weighted regression (GWR),
multiscale GWR (MGWR), and spatiotemporal weighted regression (STWR) models.

Built on the sparse generalized linear modeling (spglm) framework.
"""

__version__ = "2.1.2"

from . import gwr
from . import sel_bw
from . import diagnostics
from . import kernels
from . import search
from . import summary
from . import utils
