set -e

COVERAGE_ARGS=""
if [[ "${COVERAGE}" == "true" ]]; then
    export NUMBA_DISABLE_JIT=1
    COVERAGE_ARGS="--junitxml=junit/test-results.xml --cov=pynndescent/ --cov-report=xml --cov-report=html"
fi

pytest ${MODULE} --show-capture=no -v --disable-warnings ${COVERAGE_ARGS} && black --diff ${MODULE} && black --check ${MODULE}
