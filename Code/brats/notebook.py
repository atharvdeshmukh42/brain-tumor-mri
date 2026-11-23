"""
Notebook start-up helpers.

Uses only the standard library, so it can run before any missing package has been installed.
"""
from __future__ import annotations

import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec


def install_missing(packages: dict[str, str]) -> None:
    """Install packages that are missing; packages maps import name -> pip name (e.g. {'sklearn': 'scikit-learn'}).

    If installing changes numpy or torch, the already-loaded versions are stale, so we stop and ask for a restart.
    """
    missing = [pip_name for module, pip_name in packages.items() if find_spec(module) is None]
    if not missing:
        return

    def core_versions():
        found = {}
        for name in ('numpy', 'torch'):
            try:
                found[name] = version(name)
            except PackageNotFoundError:
                found[name] = None
        return found

    before = core_versions()
    print('Installing', missing)
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *missing])
    after = core_versions()
    if before != after:
        raise RuntimeError(f'pip changed {before} -> {after}. Click Runtime -> Restart session, then Run all again.')
