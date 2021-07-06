import pkg_resources
import numba
from .pynndescent_ import NNDescent, PyNNDescentTransformer

# Workaround: https://github.com/numba/numba/issues/3341
if numba.config.THREADING_LAYER == "omp":
    try:
        from numba.np.ufunc import tbbpool

        numba.config.THREADING_LAYER = "tbb"
    except ImportError as e:
        # might be a missing symbol due to e.g. tbb libraries missing
        numba.config.THREADING_LAYER = "workqueue"

__version__ = pkg_resources.get_distribution("pynndescent").version
