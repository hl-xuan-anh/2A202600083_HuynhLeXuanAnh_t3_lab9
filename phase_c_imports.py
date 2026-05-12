from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    # Ensure the module is visible during execution (dataclasses and others expect this).
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


_REPO_ROOT = Path(__file__).resolve().parent
_PHASE_C_DIR = _REPO_ROOT / "phase-c"

_input_guard = _load_module("phase_c_input_guard", _PHASE_C_DIR / "input_guard.py")
_topic_guard = _load_module("phase_c_topic_guard", _PHASE_C_DIR / "topic_guard.py")
_output_guard = _load_module("phase_c_output_guard", _PHASE_C_DIR / "output_guard.py")

InputGuard = getattr(_input_guard, "InputGuard")
TopicGuard = getattr(_topic_guard, "TopicGuard")
OutputGuardAPI = getattr(_output_guard, "OutputGuardAPI")

__all__ = ["InputGuard", "TopicGuard", "OutputGuardAPI"]
