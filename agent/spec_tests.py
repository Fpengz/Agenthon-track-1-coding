"""Require spec-first tests to read deliverables before using their reported evidence."""

from __future__ import annotations

import re

from agent.executor import ExecutionResult
from agent.review import VerifierResult, verifier_result

_READS = re.compile(r"^\[agent\] spec-first output reads: ([0-9]+)$", re.M)

_AUDIT = """import atexit as _agent_atexit, os as _agent_os, sys as _agent_sys
_agent_output_root = _agent_os.path.abspath(_agent_os.environ['OUTPUT_DIR']) + _agent_os.sep
_agent_output_reads = set()
def _agent_test_audit(event, args):
    if event != 'open' or not isinstance(args[0], (str, bytes)):
        return
    path, mode, flags = args
    readable = ('r' in mode or '+' in mode) if mode else flags & _agent_os.O_ACCMODE != _agent_os.O_WRONLY
    path = _agent_os.path.abspath(_agent_os.fsdecode(path))
    if readable and path.startswith(_agent_output_root):
        _agent_output_reads.add(path)
def _agent_report_reads():
    print('[agent] spec-first output reads:', len(_agent_output_reads), file=_agent_sys.stderr)
_agent_atexit.register(_agent_report_reads)
_agent_sys.addaudithook(_agent_test_audit)
"""


def audited_tests(code: str) -> str:
    """Install the read audit without changing a suite's future imports or main guard."""
    return _AUDIT + f"\nexec(compile({code!r}, '<spec-first-tests>', 'exec'))\n"


def test_evidence(run: ExecutionResult) -> VerifierResult:
    """Input-only checks cannot establish either output correctness or an output defect."""
    reads = _READS.findall(run.stderr)
    if not reads or int(reads[-1]) == 0:
        return VerifierResult(0, [], "Spec-first tests did not read any deliverable.")
    return verifier_result(run)
