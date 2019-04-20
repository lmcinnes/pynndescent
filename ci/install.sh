set -e

pip install -r requirements.txt

if [[ "${COVERAGE}" == "true" ]]; then
    pip install coverage coveralls
fi

pip install -e .
