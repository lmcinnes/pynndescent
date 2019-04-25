set -e

COVERAGE_ARGS=""
if [[ "${COVERAGE}" == "true" ]]; then
    export NUMBA_DISABLE_JIT=1
    COVERAGE_ARGS="--with-coverage --cover-package=${MODULE}"
fi

nosetests -s -v ${COVERAGE_ARGS} ${MODULE} && black --check ${MODULE}
