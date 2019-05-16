import pkg_resources
import numba
from .pynndescent_ import NNDescent, PyNNDescentTransformer

# Workaround: https://github.com/numba/numba/issues/3341
numba.config.THREADING_LAYER = "workqueue"

__version__ = pkg_resources.get_distribution("pynndescent").version
