"""Static checks before a generated script is executed (no House request, no run).

19% of execution failures in the notes runs were statically detectable (IndentationError 47,
NameError 41, SyntaxError 33, ImportError 23 of 753). Execution stops at the FIRST such error;
static analysis reports all of them at once, so one repair can fix every one. Checks:

- ``compile()``: syntax and indentation errors, with the line.
- pyflakes: undefined names only (style warnings never block).
- imports: every imported module must be installed (the provided ``agent_skills`` is fine).
- paths: string literals under ``/app/`` (other than ``/app/output``) do not exist when a
  submission runs -- they are unit-image paths; ``/input`` and ``/app/output`` do exist.
- pandas APIs removed by pandas 3 (``fillna(method=...)``, ``iteritems``, ``pd.np``).

Every check must stay silent on scripts that actually ran cleanly (validated on past runs).
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import re
import sys
import warnings

_REMOVED_PANDAS = [
    (
        re.compile(r"\.fillna\([^)]*\bmethod\s*="),
        "fillna(method=...) was removed in pandas 3: use .ffill()/.bfill()",
    ),
    (re.compile(r"\.iteritems\("), ".iteritems() was removed: use .items()"),
    (re.compile(r"\bpd\.np\."), "pd.np was removed: import numpy as np"),
]
_ALWAYS_AVAILABLE = frozenset({"agent_skills", "__future__"})


def _undefined_names(code: str) -> list[str]:
    try:
        from pyflakes import checker, messages
    except ImportError:  # host without pyflakes: skip this check
        return []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tree = ast.parse(code)
    found = []
    for msg in checker.Checker(tree, filename="_solution.py").messages:
        if isinstance(msg, (messages.UndefinedName, messages.UndefinedLocal)):
            found.append(f"line {msg.lineno}: undefined name {msg.message_args[0]!r}")
    return found


def _task_modules(task_dir: pathlib.Path | None) -> set[str]:
    """Module names a unit ships itself (``.py`` files anywhere under the task directory)."""
    if task_dir is None or not task_dir.is_dir():
        return set()
    return {p.stem for p in task_dir.rglob("*.py")}


def _missing_modules(tree: ast.Module, available: set[str]) -> list[str]:
    """Module-level imports (not inside try blocks) of modules that cannot be found."""
    missing = []
    for node in tree.body:  # module scope only: imports in functions or try blocks may not run
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        for name in names:
            top = name.split(".")[0]
            if top in _ALWAYS_AVAILABLE or top in available or top in sys.builtin_module_names:
                continue
            if importlib.util.find_spec(top) is None:
                missing.append(f"line {node.lineno}: module {top!r} is not installed (no network)")
    return missing


def _bad_paths(tree: ast.AST) -> list[str]:
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            if value.startswith("/app/") and not value.startswith("/app/output"):
                bad.append(
                    f"line {node.lineno}: {value!r} does not exist at run time (unit-image path); "
                    "read it from TASK_DIR using the path mapping in the prompt"
                )
    return bad


def preflight(code: str, task_dir: pathlib.Path | None = None) -> tuple[list[str], list[str]]:
    """``(blocking, advisory)`` findings for ``code``.

    Blocking findings are certain failures (the script is not run). Advisory findings are likely
    bugs that may sit on paths that never execute (1.7% of scripts that ran cleanly had an
    undefined name), so they only join the feedback when the run fails.
    """
    try:
        with warnings.catch_warnings():  # e.g. invalid escape sequences: harmless, not errors
            warnings.simplefilter("ignore")
            tree = ast.parse(code)
            compile(code, "_solution.py", "exec")
    except SyntaxError as exc:
        kind = exc.__class__.__name__
        text = (exc.text or "").strip()
        return [f"line {exc.lineno}: {kind}: {exc.msg}" + (f" -> {text}" if text else "")], []
    blocking = _missing_modules(tree, _task_modules(task_dir))
    advisory = _undefined_names(code) + _bad_paths(tree)
    for pattern, message in _REMOVED_PANDAS:
        for match in pattern.finditer(code):
            advisory.append(f"line {code.count(chr(10), 0, match.start()) + 1}: {message}")
    return blocking[:20], advisory[:20]
