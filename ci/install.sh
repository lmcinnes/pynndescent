set -e

pip install -r requirements.txt
pip install black

if [[ "${COVERAGE}" == "true" ]]; then
    pip install pytest-cov coveralls
fi

pip install -e .
