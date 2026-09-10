#!/usr/bin/env sh
rm -rf .venv
uv venv .venv/
uv pip install --python .venv/bin/python ipykernel
.venv/bin/python -m ipykernel install --user \
    --name radiogat \
    --display-name "Python (RadioGAT)"
.venv/bin/python -c "import ipykernel; print(ipykernel.__file__)"
