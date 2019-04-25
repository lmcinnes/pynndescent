set -e

pip install -r requirements.txt
pip install black

if [[ "${COVERAGE}" == "true" ]]; then
    pip install coverage coveralls
fi

pip install -e .
