"""Structured error context for generated scripts.

28% of execution failures repeated the previous attempt's error (753 failures in the notes
runs): a bare traceback often lacks what the model needs, e.g. which columns a DataFrame
actually has when a ``KeyError`` names the wrong one. ``HOOK_SOURCE`` is written as
``sitecustomize.py`` next to each script (Python imports it at start-up, so the script still runs
as a normal ``__main__`` with unchanged line numbers). On an uncaught exception it prints, after
the usual traceback, a compact ``[agent]`` block: the failure type, the failing script line, and
the variables in scope there (DataFrame shapes/columns/index, dict keys, array shapes/dtypes).
"""

from __future__ import annotations

HOOK_SOURCE = r"""
import linecache, sys

_ORIGINAL_HOOK = sys.excepthook
_MAX_CHARS = 1500

def _classify(exc_type, exc):
    name = exc_type.__name__
    text = str(exc).lower()
    if name == "KeyError":
        return "missing key or column (KeyError)"
    if name in ("FileNotFoundError", "IsADirectoryError"):
        return "missing file or wrong path"
    if name in ("ParserError", "JSONDecodeError", "UnicodeDecodeError", "EmptyDataError") or (
        name == "ValueError" and any(w in text for w in ("could not convert", "parse", "time data", "invalid literal"))
    ):
        return "parse error (" + name + ")"
    if name == "ValueError" and any(w in text for w in ("shape", "broadcast", "length", "dimension", "mismatch", "must be of the same")):
        return "shape or length mismatch (ValueError)"
    if name == "IndexError":
        return "index out of range (IndexError)"
    if name in ("TypeError", "AttributeError"):
        return "API or type misuse (" + name + "); check the installed library versions"
    if name in ("ImportError", "ModuleNotFoundError"):
        return "module unavailable offline (" + name + ")"
    if name in ("ZeroDivisionError", "FloatingPointError", "OverflowError"):
        return "numeric error (" + name + ")"
    if name == "MemoryError":
        return "out of memory"
    if name in ("SyntaxError", "IndentationError", "TabError"):
        return "invalid Python (" + name + ")"
    return name

def _describe(value):
    cls = type(value)
    mod, name = cls.__module__, cls.__name__
    try:
        if mod.startswith("pandas") and name == "DataFrame":
            cols = [str(c) for c in list(value.columns)[:30]]
            more = " ..." if value.shape[1] > 30 else ""
            return "DataFrame shape=%s columns=%s%s index=%s[%s]" % (
                value.shape, cols, more, type(value.index).__name__, value.index.dtype)
        if mod.startswith("pandas") and name == "Series":
            return "Series len=%d dtype=%s name=%r index=%s" % (
                len(value), value.dtype, value.name, type(value.index).__name__)
        if mod.startswith("numpy") and name == "ndarray":
            return "ndarray shape=%s dtype=%s" % (value.shape, value.dtype)
        if isinstance(value, dict) and value:
            keys = [repr(k) for k in list(value)[:25]]
            return "dict len=%d keys=[%s]%s" % (len(value), ", ".join(keys), " ..." if len(value) > 25 else "")
    except Exception:
        return None
    return None

def _hook(exc_type, exc, tb):
    _ORIGINAL_HOOK(exc_type, exc, tb)
    try:
        target = None
        while tb is not None:
            if tb.tb_frame.f_code.co_filename.endswith("_solution.py"):
                target = tb
            tb = tb.tb_next
        lines = ["[agent] failure type: " + _classify(exc_type, exc)]
        if target is not None:
            code = linecache.getline(target.tb_frame.f_code.co_filename, target.tb_lineno).strip()
            lines.append("[agent] failing line %d: %s" % (target.tb_lineno, code))
            described = 0
            for var, value in list(target.tb_frame.f_locals.items()):
                if var.startswith("_") or described >= 20:
                    continue
                text = _describe(value)
                if text:
                    lines.append("[agent]   %s: %s" % (var, text))
                    described += 1
        sys.stderr.write("\n" + "\n".join(lines)[:_MAX_CHARS] + "\n")
    except Exception:
        pass

sys.excepthook = _hook
"""
