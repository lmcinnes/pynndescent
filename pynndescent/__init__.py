from .pynndescent_ import NNDescent, PyNNDescentTransformer

# Workaround: https://github.com/numba/numba/issues/3341
import numba

numba.config.THREADING_LAYER = "workqueue"

import pkg_resources

__version__ = pkg_resources.get_distribution("pynndescent").version
