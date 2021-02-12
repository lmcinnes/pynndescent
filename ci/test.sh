set -e

COVERAGE_ARGS=""
if [[ "${COVERAGE}" == "true" ]]; then
    export NUMBA_DISABLE_JIT=1
    COVERAGE_ARGS="--cov=${MODULE}"
fi

pytest --show-capture=no -v --disable-warnings ${COVERAGE_ARGS} ${MODULE} && black --diff ${MODULE} && black --check ${MODULE}
