"""
core.vendor_loader
==================

Loads the team members' original files **exactly as they wrote them**, from
``vendor/``, without editing a single line and without requiring their
`sys.path` layout.

Why not just ``import``?  Because each team member wrote their module as a
standalone project with its own top-level names (``detection``,
``analytics``, ``src``).  Importing them normally would either need
`sys.path` surgery or a rename -- both of which mean touching their code.
``importlib`` file-loading gives us their real classes under a private,
collision-free name instead.

Everything here returns the ORIGINAL objects:
    load_vendor_module("person2_traffic/src/detection/vehicle_detector.py")
        -> the real VehicleDetector class Person 2 wrote.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict

from .logging_setup import get_logger

VENDOR_ROOT = Path(__file__).resolve().parent.parent / "vendor"

_log = get_logger("vendor")
_cache: Dict[str, ModuleType] = {}


class VendorImportError(ImportError):
    """The vendored file exists but could not be imported (usually a missing
    third-party dependency such as easyocr)."""


def vendor_path(relative: str) -> Path:
    return VENDOR_ROOT / relative


def load_vendor_module(
    relative: str,
    alias: str | None = None,
    sys_path: str | None = None,
) -> ModuleType:
    """Import ``vendor/<relative>`` as a private module object.

    Parameters
    ----------
    relative:
        Path inside ``vendor/``, e.g. ``"person2_traffic/src/analytics/congestion.py"``
    alias:
        Optional module name.  Defaults to a namespaced name derived from the
        path so it can never clash with anything else on sys.modules.
    sys_path:
        Optional directory inside ``vendor/`` to put on ``sys.path`` for the
        duration of the import.  Person 4's files do a real package import
        (``from ai.vehicle.classes import VEHICLE_CLASSES``), so their own
        project root has to be importable while their file executes.  It is
        removed again afterwards so the rest of the platform is unaffected.
    """
    key = str(relative)
    if key in _cache:
        return _cache[key]

    path = vendor_path(relative)
    if not path.exists():
        raise VendorImportError(f"vendored file not found: {path}")

    name = alias or ("uip_vendor." + key.replace("/", ".").replace("\\", ".")[:-3])

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:                      # pragma: no cover
        raise VendorImportError(f"cannot build import spec for {path}")

    extra_path = str(vendor_path(sys_path)) if sys_path else None
    added = False
    if extra_path and extra_path not in sys.path:
        sys.path.insert(0, extra_path)
        added = True

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:                                     # noqa: BLE001
        sys.modules.pop(name, None)
        raise VendorImportError(f"{path.name}: {type(exc).__name__}: {exc}") from exc
    finally:
        if added:
            try:
                sys.path.remove(extra_path)
            except ValueError:                                   # pragma: no cover
                pass

    _log.debug("loaded vendor module %s", relative)
    _cache[key] = module
    return module


def load_vendor_attr(relative: str, attribute: str):
    """Convenience: import a vendored file and pull one name out of it."""
    module = load_vendor_module(relative)
    if not hasattr(module, attribute):
        raise VendorImportError(f"{relative} has no attribute '{attribute}'")
    return getattr(module, attribute)
