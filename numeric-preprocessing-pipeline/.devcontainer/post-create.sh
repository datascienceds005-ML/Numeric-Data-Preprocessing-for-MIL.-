#!/usr/bin/env bash
# Codespaces post-create: install the package with dev tooling and verify the
# toolchain so the very first terminal is ready for `make verify`.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "==> Installing numprep with dev tooling (a few minutes on first build)…"
python -m pip install --upgrade pip --quiet
python -m pip install -e ".[dev]" --quiet

echo "==> Toolchain sanity check…"
python -c "import numprep, pandas, sklearn, polars, pyarrow, pandera; print('    imports OK')"
numprep --help > /dev/null && echo "    CLI OK"
echo "    $(ruff --version)  |  $(mypy --version)  |  pytest $(python -m pytest --version 2>/dev/null | head -1 | awk '{print $2}')"

echo
echo "Devcontainer ready. Next steps:"
echo "  make verify                        # lint + typecheck + full test suite"
echo "  make synthesize && make run-local  # end-to-end demo on seeded dirty data"
echo "  make help                          # all targets"
