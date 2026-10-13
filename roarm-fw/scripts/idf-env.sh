# Source from repo root:  . scripts/idf-env.sh
# Puts Python 3.11 ahead of system 3.14, then loads ESP-IDF.

_ROARM_PY311="${IDF_PYTHON3:-$HOME/.local/bin/python3.11}"
_ROARM_PYBIN="${XDG_CACHE_HOME:-$HOME/.cache}/roarm-fw/pybin"
mkdir -p "$_ROARM_PYBIN"
ln -sfn "$_ROARM_PY311" "$_ROARM_PYBIN/python3"
ln -sfn "$_ROARM_PY311" "$_ROARM_PYBIN/python"
export PATH="$_ROARM_PYBIN:$PATH"

# This machine's IDF venv was created with /tmp/py311bin (gone after reboot).
mkdir -p /tmp/py311bin
ln -sfn "$_ROARM_PY311" /tmp/py311bin/python3
ln -sfn "$_ROARM_PY311" /tmp/py311bin/python
unset _ROARM_PY311 _ROARM_PYBIN

. "$HOME/esp/esp-idf/export.sh"
