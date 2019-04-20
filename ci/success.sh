set -e

if [[ "${COVERAGE}" == "true" ]]; then
    coveralls || echo "Coveralls upload failed"
fi