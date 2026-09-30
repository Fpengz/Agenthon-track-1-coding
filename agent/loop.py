"""Core Agent solver loop with multi-turn self-repair."""

from __future__ import annotations

import logging
import os
import pathlib
import re
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from openai.types.chat import ChatCompletionMessageParam

from agent.client import MAX_OUTPUT_TOKENS, HouseModelClient
from agent.consensus import compare_outputs
from agent.context import canary_leaks, card_facts, sanitize_instruction, task_canaries
from agent.examples import format_example, select_example
from agent.executor import (
    apply_edits,
    extract_partial_code,
    extract_python_code,
    join_continuation,
    run_code,
    strip_reasoning,
)
from agent.inputs import build_input_previews
from agent.knowledge import domain_notes, read_card
from agent.prompts import (
    SYSTEM_PROMPT,
    VERIFY_REPAIR_HEADLINE,
    build_continuation_prompt,
    build_exploration_section,
    build_initial_prompt,
    build_judge_prompt,
    build_no_code_prompt,
    build_plan_prompt,
    build_repair_prompt,
    build_review_prompt,
    build_verify_prompt,
)
from agent.review import (
    output_diagnostics,
    output_files,
    parse_verdict,
    restore_outputs,
    snapshot_outputs,
)
from agent.skills import SKILLS_SUMMARY
from agent.spec import DeliverableSpec, parse_spec, spec_problems

logger = logging.getLogger(__name__)
NO_CODE_MARKER = "\n\n### NOTE:\n"
# Follow-up requests allowed for one script that outgrows the 4,000-token output cap.
MAX_CONTINUATIONS = 2
# Self-reviews of a clean run before it is accepted (each costs one House request).
MAX_REVIEWS = int(os.environ.get("AGENT_MAX_REVIEWS", "2"))  # 0 disables review (A/B runs)
# No new review starts after this share of the unit's time budget (480 s at the default 600 s),
# so reviews scale with AGENT_TIME_BUDGET_SEC instead of silently stopping under heavier load.
REVIEW_DEADLINE_SHARE = 0.8
EXEC_TIMEOUT_SEC = 300
# Per-unit time budget. The card's [agent].timeout_sec is a hard limit (a timeout scores 0 even
# with good outputs), and the whole roster shares one wall-clock allowance (12 h / 86 units in
# Development), so the agent stops starting attempts at the smaller of 85% of the card timeout
# and AGENT_TIME_BUDGET_SEC, and clips script timeouts to the time left.
TIME_BUDGET_SEC = float(os.environ.get("AGENT_TIME_BUDGET_SEC", "600"))
CARD_TIMEOUT_SHARE = 0.85
DEFAULT_CARD_TIMEOUT_SEC = 1800.0
MIN_ATTEMPT_SEC = 60.0  # do not start an attempt (request + run) with less time than this
MIN_EXEC_TIMEOUT_SEC = 30
# Best-of-N consensus (agent/consensus.py). AGENT_CANDIDATES=1 keeps the single-candidate loop.
# With more, candidate A (the full loop, with review) runs first, leaving a small reserve of the
# time budget; later candidates are independent solves (fresh prompt, sampled, no review, few
# attempts) into their own staging directory. A near-identical pair ends the search early;
# otherwise, with 3+ candidates the medoid (highest total agreement with the others) is
# submitted, and with 2 a judge request decides. Independent solutions rarely agree exactly
# (smoke run: pairwise agreement 0.40-0.91), so the medoid uses partial agreement as evidence.
CANDIDATES = max(1, int(os.environ.get("AGENT_CANDIDATES", "1")))
CANDIDATE_RESERVE_SHARE = 0.3
# Instruction-derived output checks (agent/spec.py): missing columns / JSON keys after a clean
# run trigger up to MAX_SPEC_REPAIRS repair turns. Off by default until A/B-tested.
SPEC_CHECKS = os.environ.get("AGENT_SPEC_CHECKS", "0").strip().lower() in {"1", "true", "on"}
MAX_SPEC_REPAIRS = 2
# Few-shot reference examples from agent/examples/library.jsonl (rule 8: OTHER units only).
EXAMPLES = os.environ.get("AGENT_EXAMPLES", "0").strip().lower() in {"1", "true", "on"}
# Model-written verification tests (executed on a COPY of the outputs) after a clean run, and a
# requirements checklist extracted up front. Both spend otherwise-unused requests on evidence
# about correctness: 115 of 136 failing units ended "accepted but wrong".
VERIFY = os.environ.get("AGENT_VERIFY", "0").strip().lower() in {"1", "true", "on"}
MAX_VERIFY = 2
VERIFY_TIMEOUT_SEC = 120
PLAN = os.environ.get("AGENT_PLAN", "0").strip().lower() in {"1", "true", "on"}
MAX_CHECKLIST_CHARS = 3000
# Tool use: up to AGENT_EXPLORE exploration snippets (run read-only, output fed back) before the
# final script. Skills: the vetted agent_skills helpers are always importable; AGENT_SKILLS=1
# advertises them in the prompt.
EXPLORE_STEPS = max(0, int(os.environ.get("AGENT_EXPLORE", "0")))
EXPLORE_TIMEOUT_SEC = 60
EXPLORE_OUTPUT_CHARS = 2500
EXPLORE_LOG_CHARS = 7000
SKILLS = os.environ.get("AGENT_SKILLS", "0").strip().lower() in {"1", "true", "on"}
_EXPLORE_BLOCK = re.compile(r"^```[ \t]*explore[^\n]*\n(.*?)^```[ \t]*$", re.S | re.M)
# One function per deliverable, failures isolated and reported together (fewer no-output units).
STRUCTURED = os.environ.get("AGENT_STRUCTURED", "0").strip().lower() in {
    "1",
    "true",
    "on",
}  # at most this share of the budget is held back from candidate A
CANDIDATE_ATTEMPTS = 6
CANDIDATE_TEMPERATURE = 0.7
CANDIDATE_MIN_SEC = 120.0
_CHOICE = re.compile(r"CHOICE:\s*\**\s*([AB])\b")
# Repairs/reviews of scripts at least this long ask for SEARCH/REPLACE edits, not a rewrite.
EDIT_MODE_MIN_LINES = 80
# Context guard. The House route documents no serving window, and an over-long request is
# rejected upstream after admission, so it still costs one of the 25 requests. The local vLLM
# serves 32,768 tokens. 2.5 chars/token sits just below the lowest ratio measured on real
# prompts (2.53; median 3.24), so estimates err toward "too big".
CONTEXT_TOKENS = int(os.environ.get("HOUSE_CONTEXT_TOKENS", "32768"))
CHARS_PER_TOKEN = 2.5
CONTEXT_MARGIN_TOKENS = 256
MIN_OUTPUT_TOKENS = 1024
# Optional task-prompt parts, dropped cumulatively in this order when a request would not fit.
_DROP_ORDER: list[frozenset[str]] = [
    frozenset(),
    frozenset({"reference_example"}),
    frozenset({"reference_example", "domain_notes"}),
    frozenset({"reference_example", "domain_notes", "requirements_checklist"}),
    frozenset({"reference_example", "domain_notes", "requirements_checklist", "input_previews"}),
]


def _estimated_tokens(user_prompt: str) -> int:
    return int((len(SYSTEM_PROMPT) + len(user_prompt)) / CHARS_PER_TOKEN)


def fit_to_context(
    user_prompt: str, task_prompt: str, variants: list[str]
) -> tuple[str, str, int] | None:
    """Make a request fit the context window: ``(user_prompt, task_prompt, max_tokens)``.

    Every turn's prompt starts with the task prompt, so a leaner task-prompt variant can be
    swapped in (and kept for later turns). If even the leanest does not fit with the full output
    cap, the output cap shrinks; ``None`` if not even ``MIN_OUTPUT_TOKENS`` would remain.
    """
    budget = CONTEXT_TOKENS - MAX_OUTPUT_TOKENS - CONTEXT_MARGIN_TOKENS
    if _estimated_tokens(user_prompt) <= budget:
        return user_prompt, task_prompt, MAX_OUTPUT_TOKENS
    suffix = user_prompt[len(task_prompt) :] if user_prompt.startswith(task_prompt) else None
    if suffix is not None:
        start = variants.index(task_prompt) + 1 if task_prompt in variants else len(variants)
        for variant in variants[start:]:
            task_prompt, user_prompt = variant, variant + suffix
            if _estimated_tokens(user_prompt) <= budget:
                logger.warning(
                    "Prompt too large for the context window; using a leaner task prompt "
                    "(~%d tokens)",
                    _estimated_tokens(user_prompt),
                )
                return user_prompt, task_prompt, MAX_OUTPUT_TOKENS
    room = CONTEXT_TOKENS - CONTEXT_MARGIN_TOKENS - _estimated_tokens(user_prompt)
    if room < MIN_OUTPUT_TOKENS:
        return None
    logger.warning("Prompt near the context limit; reducing the output cap to %d tokens", room)
    return user_prompt, task_prompt, min(room, MAX_OUTPUT_TOKENS)


def _edit_mode(code: str) -> bool:
    return len(code.splitlines()) >= EDIT_MODE_MIN_LINES


def _review_findings(response: str) -> str:
    """The problem list from a review reply that was cut off before its corrected script."""
    answer = response.split("</think>")[-1]
    tail = answer.split("VERDICT", 1)[-1] if "VERDICT" in answer else answer
    return f"Review findings:\n{tail.strip()[:4000]}"


# Graded answers in a local unit checkout; never mounted at evaluation (scripts/build_dev_dataset.py).
ANSWER_DIRS = frozenset({"checks", "reference"})
DATA_SUFFIXES = frozenset(
    {".parquet", ".pqt", ".csv", ".tsv", ".json", ".jsonl", ".h5", ".pkl", ".txt", ".xlsx", ".xml"}
)
# Deliverables named as `/output/<name>` in an instruction (charts and reports included).
OUTPUT_REF = re.compile(
    r"/output/([\w\-]+\.(?:parquet|csv|tsv|jsonl?|html|png|svg|pdf|txt|md|xlsx|npy))",
    re.IGNORECASE,
)


def container_path_map(task_dir: pathlib.Path) -> list[tuple[str, str]]:
    """Map image paths an instruction may cite (e.g. ``/app/data/x.parquet``) to task files.

    Unit instructions describe the unit image, whose ``environment/Dockerfile`` COPYs data
    (sometimes renamed) into ``/app``. A submission never runs in that image: it gets the raw
    unit at ``/input``. Resolving the COPY lines tells the model which task file each cited
    path is. Returns ``(container_path, path relative to task_dir)`` pairs.
    """
    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.is_file():
        return []
    context = task_dir / "environment"
    mapping: list[tuple[str, str]] = []
    for line in dockerfile.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 3 or parts[0].upper() not in {"COPY", "ADD"}:
            continue
        args = [a for a in parts[1:] if not a.startswith("--")]
        *sources, dest = args
        for src in sources:
            src_path = context / src
            if src_path.is_dir():
                for f in sorted(p for p in src_path.rglob("*") if p.is_file()):
                    target = f"{dest.rstrip('/')}/{f.relative_to(src_path).as_posix()}"
                    mapping.append((target, f.relative_to(task_dir).as_posix()))
            elif src_path.is_file():
                into_dir = dest.endswith("/") or len(sources) > 1
                target = f"{dest.rstrip('/')}/{src_path.name}" if into_dir else dest
                mapping.append((target, src_path.relative_to(task_dir).as_posix()))
    return mapping


@dataclass(frozen=True)
class _PromptParts:
    """Everything the task prompt is built from, so it can be rebuilt for another OUTPUT_DIR."""

    instruction_text: str
    discovered_files: list[str]
    path_map: list[tuple[str, str]]
    optional: dict[str, str]  # dropped in _DROP_ORDER when a request would not fit
    task_facts: str = ""


class AgentSolver:
    """Manages the iterative solving, execution, and self-repair loop for a task."""

    def __init__(
        self,
        task_dir: pathlib.Path,
        out_dir: pathlib.Path,
        max_retries: int = 15,
        client: HouseModelClient | None = None,
    ) -> None:
        self.task_dir = pathlib.Path(task_dir).resolve()
        self.out_dir = pathlib.Path(out_dir).resolve()
        self.max_retries = max_retries
        self.client = client or HouseModelClient()
        self.spec: list[DeliverableSpec] = []
        self.canaries: set[str] = set()
        self.accepted_code = ""
        self.stop_event = threading.Event()  # set when parallel candidates reach consensus
        self.time_budget = self._time_budget()
        self.deadline = time.monotonic() + self.time_budget
        logger.debug(
            "Configured AgentSolver (task_dir=%s, out_dir=%s, max_retries=%d)",
            self.task_dir,
            self.out_dir,
            self.max_retries,
        )

    def _time_budget(self) -> float:
        card = read_card(self.task_dir).get("agent", {})
        try:
            card_timeout = float(card.get("timeout_sec", DEFAULT_CARD_TIMEOUT_SEC))
        except (TypeError, ValueError):
            card_timeout = DEFAULT_CARD_TIMEOUT_SEC
        return min(CARD_TIMEOUT_SHARE * card_timeout, TIME_BUDGET_SEC)

    def _time_left(self) -> float:
        return self.deadline - time.monotonic()

    def discover_inputs(self) -> list[pathlib.Path]:
        """Discover the input files supplied with the task.

        Everything under ``environment/`` (except its Dockerfile) is task data, whatever its
        format (.pqt, .tsv, .xlsx, .xml, .zip, ...); elsewhere only data-like files count.
        """
        data_paths: list[pathlib.Path] = []
        for p in sorted(self.task_dir.rglob("*")):
            rel = p.relative_to(self.task_dir)
            # A local unit checkout also carries graded answers; they are never
            # mounted at evaluation and must not reach the prompt.
            if not p.is_file() or ANSWER_DIRS.intersection(rel.parts[:1]):
                continue
            if "__pycache__" in rel.parts or p.name in {"card.toml", "instruction.md"}:
                continue
            if rel.parts[0] == "environment":
                if p.name != "Dockerfile":
                    data_paths.append(p)
            elif p.suffix.lower() in DATA_SUFFIXES:
                data_paths.append(p)
        logger.info(
            "Input discovery found %d supported files: %s",
            len(data_paths),
            [
                f"{path.relative_to(self.task_dir)} ({path.stat().st_size} bytes)"
                for path in data_paths
            ],
        )
        return data_paths

    def find_instruction(self) -> str:
        """Read instruction.md from the task directory."""
        candidates = [
            self.task_dir / "instruction.md",
            self.task_dir / "input" / "instruction.md",
        ]
        for c in candidates:
            if c.is_file():
                instruction = c.read_text(encoding="utf-8")
                logger.info(
                    "Loaded task instruction from %s (%d characters)",
                    c,
                    len(instruction),
                )
                return instruction
        raise FileNotFoundError(f"instruction.md not found in {self.task_dir}")

    def detect_expected_deliverables(
        self, instruction_text: str, input_files: list[pathlib.Path] | None = None
    ) -> list[str]:
        """Heuristically extract mentioned deliverable filenames from instruction."""
        input_names = {p.name for p in input_files} if input_files else set()
        # Exclude common metadata/benchmark files and input files
        ignore = {"card.toml", "manifest.json", "reward.json", "pytest_report.json"} | input_names
        # Prefer explicit `/output/<name>` references; fall back to any data filename.
        explicit = OUTPUT_REF.findall(instruction_text)
        matches = explicit or re.findall(
            r"[\w\-]+\.(?:parquet|csv|json)", instruction_text, re.IGNORECASE
        )
        deliverables = sorted({m for m in matches if m not in ignore})
        return deliverables

    def clear_stale_deliverables(self, expected_names: list[str]) -> None:
        """Remove expected deliverables left over from earlier attempts or runs."""
        for name in expected_names:
            path = self.out_dir / name
            if path.is_file():
                logger.info("Removing stale deliverable before execution: %s", path)
                path.unlink()

    def check_deliverables(
        self, expected_names: list[str], produced_after: float = 0.0
    ) -> tuple[bool, list[str]]:
        """Verify that deliverables exist, are non-empty, and were written by this attempt."""
        missing = []
        if expected_names:
            for name in expected_names:
                p = self.out_dir / name
                if not p.exists() or p.stat().st_size == 0 or p.stat().st_mtime < produced_after:
                    missing.append(name)
            logger.info(
                "Deliverable check: expected=%s, missing=%s",
                expected_names,
                missing,
            )
            return len(missing) == 0, missing

        # Fallback: check if any valid deliverable file exists
        valid_files = [
            p
            for p in self.out_dir.iterdir()
            if p.is_file()
            and p.name not in {"_solution.py", "reward.json", "pytest_report.json"}
            and p.suffix.lower() in DATA_SUFFIXES | {".html", ".png", ".svg", ".pdf", ".md"}
            and p.stat().st_size > 0
            and p.stat().st_mtime >= produced_after
        ]
        if not valid_files:
            logger.warning("Deliverable check found no non-empty output files in %s", self.out_dir)
            return False, ["(any deliverable file)"]
        logger.info("Deliverable check found output files: %s", [path.name for path in valid_files])
        return True, []

    def run(self) -> bool:
        """Run the end-to-end agent loop."""
        logger.info("Starting AgentSolver for task: %s", self.task_dir.name)
        logger.info("Task directory: %s", self.task_dir)
        logger.info("Deliverable directory: %s", self.out_dir)
        logger.info("Repair attempts available: %d", self.max_retries)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        instruction_text = self.find_instruction()
        input_files = self.discover_inputs()
        expected_deliverables = self.detect_expected_deliverables(
            instruction_text=instruction_text, input_files=input_files
        )

        logger.info(
            "Discovered %d input files, expecting deliverables: %s",
            len(input_files),
            expected_deliverables or "auto-detect",
        )

        relative_inputs = [str(p.relative_to(self.task_dir)) for p in input_files]
        path_map = container_path_map(self.task_dir)
        if path_map:
            logger.info("Resolved %d container path(s) from environment/Dockerfile", len(path_map))
        # The model never sees the task's canary: it is stripped from the instruction, and every
        # clean run's deliverables are scanned for it before they can be accepted.
        self.canaries = task_canaries(self.task_dir, instruction_text)
        instruction_text = sanitize_instruction(instruction_text, self.canaries)
        self.spec = parse_spec(instruction_text, expected_deliverables) if SPEC_CHECKS else []
        self._prompt_parts = _PromptParts(
            instruction_text=instruction_text,
            discovered_files=relative_inputs,
            path_map=path_map,
            task_facts=card_facts(self.task_dir),
            optional={
                "requirements_checklist": "",
                "input_previews": build_input_previews(self.task_dir, input_files),
                "domain_notes": domain_notes(self.task_dir, instruction_text),
                "reference_example": format_example(
                    select_example(self.task_dir, instruction_text) if EXAMPLES else None
                ),
            },
        )
        if PLAN:
            self._prompt_parts.optional["requirements_checklist"] = self._requirements_checklist()
        prompt_variants = self._prompt_variants(self.out_dir)
        initial_prompt = prompt_variants[0]
        logger.info(
            "Prepared initial prompt (system_chars=%d, user_chars=%d)",
            len(SYSTEM_PROMPT),
            len(initial_prompt),
        )

        with tempfile.TemporaryDirectory(prefix="agent-work-") as work_dir:
            if CANDIDATES > 1:
                return self._solve_with_consensus(expected_deliverables, pathlib.Path(work_dir))
            return self._solve_loop(
                prompt_variants=prompt_variants,
                expected_deliverables=expected_deliverables,
                work_dir=pathlib.Path(work_dir),
            )

    def _prompt_variants(self, out_dir: pathlib.Path) -> list[str]:
        """Task prompts from richest to leanest; the context guard falls back along this list."""
        parts = self._prompt_parts
        return [
            build_initial_prompt(
                instruction_text=parts.instruction_text,
                task_dir=str(self.task_dir),
                out_dir=str(out_dir),
                discovered_files=parts.discovered_files,
                path_map=parts.path_map,
                structured=STRUCTURED,
                task_facts=parts.task_facts,
                skills_summary=SKILLS_SUMMARY if SKILLS else "",
                explore_steps=EXPLORE_STEPS,
                **{k: v for k, v in parts.optional.items() if k not in dropped},
            )
            for dropped in _DROP_ORDER
        ]

    def _solve_loop(
        self,
        prompt_variants: list[str],
        expected_deliverables: list[str],
        work_dir: pathlib.Path,
        temperature: float = 0.0,
        max_reviews: int = MAX_REVIEWS,
        max_attempts: int | None = None,
    ) -> bool:
        """Generate, run and repair until a clean run is accepted; False if none was."""
        initial_prompt = prompt_variants[0]
        attempts = max_attempts or self.max_retries + 1
        # The conversation is rebuilt each turn instead of accumulated: the latest repair
        # prompt already carries the previous code and error, and replaying the model's
        # inline reasoning would overflow the House model's 32k context within a few turns.
        user_prompt = initial_prompt
        partial_code = ""  # prefix of a script cut off at the output cap, while continuing it
        continuations = 0
        reviews = 0
        reviewing = False  # the pending request is a review of a clean run
        accepted_dir = work_dir / "accepted"  # snapshot of the latest clean run's outputs
        have_accepted = False
        accepted_seconds = 0.0  # runtime of the accepted script
        reviewed_code = ""
        last_error = ""
        last_failed_code = ""
        last_feedback = ""
        edit_base = ""  # the script shown in the pending repair/review prompt (edits apply to it)
        spec_repairs = 0
        verifies = 0
        explore_steps = 0
        exploration_log: list[tuple[str, str]] = []
        verify_code = ""  # the script a verifier-driven repair was asked about
        loop_started = time.monotonic()

        # Iterative execution & self-repair loop
        try:
            for attempt in range(1, attempts + 1):
                if self.stop_event.is_set():
                    logger.info("Stopping before attempt %d: consensus already reached", attempt)
                    break
                if self._time_left() < MIN_ATTEMPT_SEC:
                    logger.warning("Time budget exhausted before attempt %d; stopping", attempt)
                    break
                logger.info("Attempt %d/%d: Calling House Model...", attempt, attempts)
                messages: list[ChatCompletionMessageParam] = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
                try:
                    fitted = fit_to_context(user_prompt, initial_prompt, prompt_variants)
                    if fitted is None:
                        logger.error("Request cannot fit the model context window; stopping")
                        break
                    user_prompt, initial_prompt, max_tokens = fitted
                    messages[1]["content"] = user_prompt
                    phase = (
                        "review"
                        if reviewing
                        else "continue"
                        if partial_code
                        else "repair"
                        if edit_base
                        else "generate"
                    )
                    result = self.client.chat(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        phase=phase,
                    )
                except Exception:
                    logger.exception("Failed to query the model on attempt %d", attempt)
                    break

                code = extract_python_code(result.content)
                explore = (
                    _EXPLORE_BLOCK.search(strip_reasoning(result.content))
                    if EXPLORE_STEPS and not code and phase == "generate" and not reviewing
                    else None
                )
                if explore and explore_steps < EXPLORE_STEPS:
                    # Tool use: run the snippet read-only and hand its output back.
                    explore_steps += 1
                    snippet = explore.group(1).strip()
                    exploration_log.append((snippet, self._explore(snippet, work_dir)))
                    while (
                        len(exploration_log) > 1
                        and sum(len(a) + len(b) for a, b in exploration_log) > EXPLORE_LOG_CHARS
                    ):
                        exploration_log.pop(0)
                    logger.info("Exploration step %d/%d", explore_steps, EXPLORE_STEPS)
                    user_prompt = initial_prompt + build_exploration_section(
                        exploration_log, EXPLORE_STEPS - explore_steps
                    )
                    continue
                if edit_base:
                    edited, unmatched = apply_edits(edit_base, result.content)
                    if edited is not None:
                        logger.info(
                            "Applied edit blocks to the %d-line script", len(edit_base.splitlines())
                        )
                        code = edited
                    elif unmatched and not code:
                        logger.warning("Edit blocks did not match the script: %s", unmatched[:3])
                        user_prompt = (
                            f"{user_prompt.split(NO_CODE_MARKER)[0]}{NO_CODE_MARKER}Your edit "
                            "blocks could not be applied (no edit was made). Resend ALL edits, "
                            "fixing these: " + "; ".join(unmatched[:5])
                        )
                        continue
                if reviewing:
                    reviewing = False
                    verdict = parse_verdict(result.content)
                    cut_off_fix = result.truncated and bool(extract_partial_code(result.content))
                    logger.info("Review verdict: %s (corrected script: %s)", verdict, bool(code))
                    if verdict == "PASS" or (verdict is None and not (code or cut_off_fix)):
                        return self._accept(attempt, "review passed", reviewed_code)
                    if not (code or cut_off_fix):
                        # FAIL, but the reply ran out of tokens before the corrected script: turn the
                        # review's findings into an ordinary repair turn instead of accepting.
                        logger.warning("Review failed without a corrected script; requesting a fix")
                        edit_base = reviewed_code
                        user_prompt = build_repair_prompt(
                            edit_mode=_edit_mode(reviewed_code),
                            task_prompt=initial_prompt,
                            previous_code=reviewed_code,
                            error_message=_review_findings(result.content),
                            out_dir=str(self.out_dir),
                            headline="Your previous script ran, but a review of its outputs against "
                            "the specification found the problems below. Fix them.",
                        )
                        continue
                    logger.warning("Review found problems; running the corrected script")
                if partial_code:
                    # This request continued a cut-off script: stitch the pieces together.
                    piece = code or (
                        extract_partial_code(result.content) if result.truncated else ""
                    )
                    if code:
                        code = join_continuation(partial_code, code)
                        partial_code = ""
                    elif piece and continuations < MAX_CONTINUATIONS:
                        partial_code = join_continuation(partial_code, piece)
                        continuations += 1
                        logger.warning(
                            "Continuation was cut off again; continuing (%d lines so far)",
                            len(partial_code.splitlines()),
                        )
                        user_prompt = build_continuation_prompt(initial_prompt, partial_code)
                        continue
                    else:
                        partial_code = ""
                if not code:
                    logger.warning(
                        "No complete Python code block in response (%d characters, finish_reason=%s)",
                        len(result.content),
                        result.finish_reason,
                    )
                    partial = extract_partial_code(result.content) if result.truncated else ""
                    if partial and continuations < MAX_CONTINUATIONS:
                        # The script itself outgrew the output cap: ask for the rest of it rather
                        # than a shorter rewrite that would be cut off at the same place.
                        partial_code = partial
                        continuations += 1
                        logger.info(
                            "Script cut off after %d lines; requesting a continuation",
                            len(partial.splitlines()),
                        )
                        user_prompt = build_continuation_prompt(initial_prompt, partial_code)
                        temperature = 0.0
                        continue
                    # Re-ask from the same base prompt. A response cut off mid-reasoning is still
                    # useful analysis: hand it back so the model writes code instead of re-deriving
                    # it (re-asking from scratch just truncates again). Sample so the retry differs.
                    user_prompt = (
                        f"{user_prompt.split(NO_CODE_MARKER)[0]}{NO_CODE_MARKER}"
                        f"{build_no_code_prompt(truncated=result.truncated, notes=result.content)}"
                    )
                    temperature = 0.6
                    continue
                temperature = 0.0
                if verify_code and code == verify_code:
                    # The model judged the verifier wrong and kept its script: the outputs of
                    # that clean run are still in place (the verifier ran on a copy).
                    return self._accept(attempt, "kept after verification", reviewed_code)
                verify_code = ""
                if code == last_failed_code:
                    # The model copied the failing script back verbatim (copying stays near-
                    # deterministic even when sampling): it cannot see the bug. Running it again
                    # only repeats the error, so ask for a fresh script WITHOUT showing the old one.
                    logger.warning(
                        "Model returned the previous failing script unchanged; re-asking"
                    )
                    edit_base = ""
                    user_prompt = (
                        f"{initial_prompt}{NO_CODE_MARKER}An earlier script for this task failed "
                        f"repeatedly with this error:\n```\n{last_feedback}\n```\nWrite a new "
                        "complete script from scratch that avoids the construct that caused it."
                    )
                    temperature = 0.7
                    continue

                logger.info(
                    "Extracted generated solution on attempt %d (%d characters, %d lines)",
                    attempt,
                    len(code),
                    len(code.splitlines()),
                )

                # Execute code in a scratch directory; deliverables go to out_dir.
                self.clear_stale_deliverables(expected_deliverables)
                started_at = time.time()
                exec_result = run_code(
                    code=code,
                    work_dir=work_dir,
                    target_out_dir=self.out_dir,
                    task_dir=self.task_dir,
                    # Once a clean run exists, a rewrite that runs far longer is a regression, not
                    # a fix; cap it so it cannot burn the roster's shared wall-clock allowance.
                    timeout_sec=max(
                        MIN_EXEC_TIMEOUT_SEC,
                        round(
                            min(
                                EXEC_TIMEOUT_SEC,
                                max(60.0, 10 * accepted_seconds)
                                if have_accepted
                                else EXEC_TIMEOUT_SEC,
                                self._time_left() - 10,
                            )
                        ),
                    ),
                )

                # Check output files
                deliverables_ok, missing = self.check_deliverables(
                    expected_deliverables, produced_after=started_at
                )

                leaks = (
                    canary_leaks(self.out_dir, self.canaries)
                    if exec_result.success and deliverables_ok
                    else []
                )
                if leaks:
                    # A deliverable carries the task's canary: that disqualifies the attempt
                    # (g2), so these outputs are never snapshotted or accepted. Put the last
                    # clean outputs back (or clear them) and repair.
                    logger.warning("Canary leaked into deliverables: %s", leaks)
                    if have_accepted:
                        restore_outputs(accepted_dir, self.out_dir)
                    else:
                        for path in output_files(self.out_dir):
                            path.unlink()
                    continuations = 0
                    last_failed_code = code
                    edit_base = code
                    user_prompt = build_repair_prompt(
                        edit_mode=_edit_mode(code),
                        task_prompt=initial_prompt,
                        previous_code=code,
                        error_message="\n".join(
                            f"{name}: contains a task identifier (a GUID from the task "
                            "metadata). Deliverables must contain only the requested results."
                            for name in leaks
                        ),
                        out_dir=str(self.out_dir),
                        headline="Your script ran, but a deliverable contains a forbidden "
                        "task identifier.",
                    )
                    continue

                if exec_result.success and deliverables_ok:
                    snapshot_outputs(self.out_dir, accepted_dir)
                    have_accepted = True
                    accepted_seconds = exec_result.duration
                    reviewed_code = code
                    last_error = ""
                    problems = spec_problems(self.spec, self.out_dir) if self.spec else []
                    if problems and spec_repairs < MAX_SPEC_REPAIRS and self._requests_left() >= 2:
                        # Clean run, but required columns/keys are missing: a precise repair turn
                        # (the imperfect run stays snapshotted as the fallback).
                        spec_repairs += 1
                        logger.warning("Spec check failed: %s", problems[:3])
                        continuations = 0
                        edit_base = code
                        user_prompt = build_repair_prompt(
                            edit_mode=_edit_mode(code),
                            task_prompt=initial_prompt,
                            previous_code=code,
                            error_message="\n".join(problems),
                            out_dir=str(self.out_dir),
                            headline="Your script ran, but its outputs do not match the "
                            "specification: required columns or keys are missing.",
                        )
                        continue
                    if (
                        VERIFY
                        and verifies < MAX_VERIFY
                        and self._requests_left() >= 3
                        and self._time_left() > 2 * MIN_ATTEMPT_SEC
                    ):
                        verifies += 1
                        report = self._run_verifier(initial_prompt, code, work_dir, accepted_dir)
                        if report:
                            logger.warning(
                                "Verification failed (round %d): %s", verifies, report[:200]
                            )
                            continuations = 0
                            edit_base = code
                            verify_code = code
                            user_prompt = build_repair_prompt(
                                edit_mode=_edit_mode(code),
                                task_prompt=initial_prompt,
                                previous_code=code,
                                error_message=report,
                                out_dir=str(self.out_dir),
                                headline=VERIFY_REPAIR_HEADLINE,
                            )
                            continue
                    elapsed = time.monotonic() - loop_started
                    requests_left = getattr(self.client, "max_requests", 0) - getattr(
                        self.client, "request_count", 0
                    )
                    review_deadline = REVIEW_DEADLINE_SHARE * self.time_budget
                    if reviews >= max_reviews or elapsed > review_deadline or requests_left < 2:
                        return self._accept(attempt, "no review budget left", reviewed_code)
                    # Exit code 0 and present files say nothing about correctness: ask the model
                    # to verify the outputs against the specification before accepting them.
                    findings = output_diagnostics(self.out_dir)
                    reviews += 1
                    reviewing = True
                    continuations = 0
                    edit_base = code
                    user_prompt = build_review_prompt(
                        edit_mode=_edit_mode(code),
                        task_prompt=initial_prompt,
                        code=code,
                        stdout=exec_result.stdout,
                        output_previews=build_input_previews(
                            self.out_dir, output_files(self.out_dir), label="OUTPUT_DIR"
                        ),
                        findings=findings,
                    )
                    logger.info(
                        "Clean run on attempt %d; requesting review %d/%d (%d automatic findings)",
                        attempt,
                        reviews,
                        max_reviews,
                        len(findings),
                    )
                    continue

                if have_accepted:
                    # A failed rewrite deleted the accepted deliverables before running: put them
                    # back now, so an external kill (card timeout) never leaves the output empty.
                    restore_outputs(accepted_dir, self.out_dir)

                # If failed, prepare feedback for the self-repair loop
                logger.warning(
                    "Attempt %d failed (returncode=%d, deliverables_ok=%s, missing=%s). Initiating repair.",
                    attempt,
                    exec_result.returncode,
                    deliverables_ok,
                    missing,
                )

                if exec_result.success and missing:
                    err_msg = (
                        "Failure type: missing deliverable -- the script exited with code 0 but "
                        f"did not write {missing} to OUTPUT_DIR ({self.out_dir}).\n"
                        f"{exec_result.feedback}"
                    )
                else:
                    err_msg = f"Failure type: {exec_result.failure_kind}\n{exec_result.feedback}"

                continuations = 0
                last_failed_code = code
                last_feedback = exec_result.feedback[-2000:]
                signature = exec_result.error_signature
                repeated = bool(signature) and signature == last_error
                last_error = signature
                if repeated:
                    logger.warning("Same error as the previous attempt: %s", signature[:120])
                edit_base = code
                user_prompt = build_repair_prompt(
                    edit_mode=_edit_mode(code),
                    task_prompt=initial_prompt,
                    repeated_error=repeated,
                    previous_code=code,
                    error_message=err_msg,
                    out_dir=str(self.out_dir),
                    missing_deliverables=missing if not deliverables_ok else None,
                )
                logger.info(
                    "Prepared repair prompt for attempt %d (error_chars=%d, prompt_chars=%d)",
                    attempt + 1,
                    len(err_msg),
                    len(user_prompt),
                )
        except Exception:
            # Never let an unexpected error lose accepted outputs or skip the fallback below.
            logger.exception("Unexpected error in the solve loop")

        if have_accepted:
            # A review "fix" failed to produce a clean run: fall back to the last clean outputs.
            restore_outputs(accepted_dir, self.out_dir)
            return self._accept(attempts, "restored the last clean run", reviewed_code)
        logger.error("Agent run stopped without producing all verified deliverables.")
        logger.info(
            "Agent run summary: status=failed, model_requests=%s, output_dir=%s",
            getattr(self.client, "request_count", "unknown"),
            self.out_dir,
        )
        return False

    def _accept(self, attempt: int, reason: str, code: str) -> bool:
        self.accepted_code = code
        self._save_solution(code)
        logger.info(
            "Task completed on attempt %d (%s; verified_outputs=%s, model_requests=%s, "
            "output_dir=%s)",
            attempt,
            reason,
            [p.name for p in output_files(self.out_dir)],
            getattr(self.client, "request_count", "unknown"),
            self.out_dir,
        )
        return True

    def _save_solution(self, code: str) -> None:
        # With transcripts on (local batch runs), keep the accepted script: checker-passed ones
        # become few-shot examples for OTHER units (agent.main build-examples).
        transcript_dir = os.environ.get("AGENT_TRANSCRIPT_DIR")
        if transcript_dir:
            try:
                path = pathlib.Path(transcript_dir)
                path.mkdir(parents=True, exist_ok=True)
                (path / "solution.py").write_text(code, encoding="utf-8")
            except OSError:
                logger.debug("Could not save the accepted script", exc_info=True)

    # ------------------------------------------------------------------ extra evidence

    def _explore(self, snippet: str, work_dir: pathlib.Path) -> str:
        """Run an exploration snippet; its output (bounded) for the next prompt.

        OUTPUT_DIR points at a scratch directory, so exploration can never create or change
        deliverables.
        """
        run = run_code(
            snippet,
            work_dir / "explore",
            work_dir / "explore-out",
            task_dir=self.task_dir,
            timeout_sec=int(min(EXPLORE_TIMEOUT_SEC, max(20.0, self._time_left() - 60))),
        )
        text = run.stdout.strip()
        if run.returncode != 0:
            text = f"{text}\n[exit {run.returncode}]\n{run.feedback}".strip()
        return text[-EXPLORE_OUTPUT_CHARS:] or "(no output)"

    def _requirements_checklist(self) -> str:
        """One request: a numbered requirements checklist extracted from the specification."""
        prompt = build_plan_prompt(self._prompt_variants(self.out_dir)[0])
        try:
            result = self.client.chat(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                phase="plan",
            )
        except Exception:
            logger.exception("Requirements checklist request failed; continuing without it")
            return ""
        text = result.content.split("</think>")[-1].replace("```", "").strip()
        logger.info("Requirements checklist: %d lines", len(text.splitlines()))
        return text[:MAX_CHECKLIST_CHARS]

    def _run_verifier(
        self, task_prompt: str, code: str, work_dir: pathlib.Path, accepted_dir: pathlib.Path
    ) -> str:
        """Model-written verification of a clean run; the failure report, or "" if it passed.

        The verifier runs against a COPY of the accepted outputs, so it cannot alter the
        deliverables. A verifier that crashes without reporting any FAIL is ignored: only
        explicit, executed failures reach the repair loop.
        """
        prompt = build_verify_prompt(
            task_prompt,
            code,
            build_input_previews(self.out_dir, output_files(self.out_dir), label="OUTPUT_DIR"),
        )
        variants = self._prompt_variants(self.out_dir)
        fitted = fit_to_context(prompt, task_prompt, variants)
        if fitted is None:
            return ""
        try:
            result = self.client.chat(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": fitted[0]},
                ],
                temperature=0.0,
                max_tokens=fitted[2],
                phase="verify",
            )
        except Exception:
            logger.exception("Verifier request failed; skipping verification")
            return ""
        verifier = extract_python_code(result.content)
        if not verifier:
            return ""
        copy_dir = work_dir / "verify-outputs"
        if copy_dir.exists():
            shutil.rmtree(copy_dir)
        shutil.copytree(accepted_dir, copy_dir)
        run = run_code(
            verifier,
            work_dir / "verify",
            copy_dir,
            task_dir=self.task_dir,
            timeout_sec=int(min(VERIFY_TIMEOUT_SEC, max(30.0, self._time_left() - 30))),
        )
        failures = [ln for ln in run.stdout.splitlines() if ln.strip().startswith("FAIL")]
        logger.info(
            "Verifier: %d PASS, %d FAIL (returncode=%d)",
            sum(ln.strip().startswith("PASS") for ln in run.stdout.splitlines()),
            len(failures),
            run.returncode,
        )
        if not failures:
            return ""
        return "Verifier failures:\n" + "\n".join(failures[:20])

    # ------------------------------------------------------------------ best-of-N consensus

    def _requests_left(self) -> int:
        return getattr(self.client, "max_requests", 0) - getattr(self.client, "request_count", 0)

    def _judge(
        self, a: tuple[str, str, pathlib.Path], b: tuple[str, str, pathlib.Path], diffs: list[str]
    ) -> tuple[str, str, pathlib.Path]:
        """One request: which of two disagreeing candidates follows the spec. Defaults to ``a``."""
        if self._requests_left() < 1 or self._time_left() < MIN_ATTEMPT_SEC:
            return a
        variants = self._prompt_variants(self.out_dir)
        prompt = build_judge_prompt(variants[0], diffs, a[1], b[1])
        fitted = fit_to_context(prompt, variants[0], variants)
        if fitted is None or fitted[2] < MAX_OUTPUT_TOKENS:
            # Too long with both scripts: judge on the disagreeing values alone.
            fitted = fit_to_context(
                build_judge_prompt(variants[0], diffs, "", ""), variants[0], variants
            )
        if fitted is None:
            return a
        try:
            result = self.client.chat(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": fitted[0]},
                ],
                temperature=0.0,
                max_tokens=fitted[2],
                phase="judge",
            )
        except Exception:
            logger.exception("Consensus judge request failed; keeping candidate %s", a[0])
            return a
        choices = _CHOICE.findall(result.content)
        choice = b if choices and choices[-1] == "B" else a
        logger.info("Consensus judge chose candidate %s (%s vs %s)", choice[0], a[0], b[0])
        return choice

    def _child(self, label: str, work_dir: pathlib.Path) -> AgentSolver:
        """An independent candidate solver sharing the client, budget, deadline and prompt parts.

        It writes to its own staging OUTPUT_DIR (named in its prompt), so no candidate can touch
        the real deliverables or another candidate's outputs.
        """
        staging = work_dir / f"out-{label}"
        staging.mkdir(parents=True, exist_ok=True)
        child = AgentSolver(
            self.task_dir, staging, max_retries=self.max_retries, client=self.client
        )
        child.deadline, child.time_budget = self.deadline, self.time_budget
        child.spec, child.canaries = self.spec, self.canaries
        child._prompt_parts = self._prompt_parts
        child.stop_event = self.stop_event
        return child

    @staticmethod
    def _run_child(
        label: str, child: AgentSolver, expected: list[str], work_dir: pathlib.Path
    ) -> bool:
        try:
            return child._solve_loop(
                child._prompt_variants(child.out_dir),
                expected,
                work_dir / f"work-{label}",
                # A is the regular loop (greedy, with review); the others are sampled and
                # unreviewed, which keeps them independent and bounds their requests.
                temperature=0.0 if label == "A" else CANDIDATE_TEMPERATURE,
                max_reviews=MAX_REVIEWS if label == "A" else 0,
                max_attempts=None if label == "A" else CANDIDATE_ATTEMPTS,
            )
        except Exception:
            logger.exception("Consensus: candidate %s crashed", label)
            return False

    def _solve_with_consensus(self, expected: list[str], work_dir: pathlib.Path) -> bool:
        """Best-of-N in PARALLEL: candidates solve concurrently as separate House requests.

        The request budget is what goes unused (failing units averaged 7.7 of 25; 115 of 136
        ended "accepted but wrong"), while the sequential version cost 521 s per unit. Running
        candidates side by side spends requests, not wall-clock time. The first agreeing pair
        stops the rest; otherwise the medoid (3+ clean) or a judge (2 clean) decides.
        """
        labels: list[str] = list("ABCDE"[:CANDIDATES])
        children: dict[str, AgentSolver] = {label: self._child(label, work_dir) for label in labels}
        clean: list[str] = []
        agreement: dict[tuple[str, str], float] = {}
        decided: str | None = None
        with ThreadPoolExecutor(max_workers=len(labels)) as pool:
            futures = {
                pool.submit(self._run_child, label, child, expected, work_dir): label
                for label, child in children.items()
            }
            for future in as_completed(futures):
                label = futures[future]
                if not future.result():
                    logger.info("Consensus: candidate %s produced no clean run", label)
                    continue
                for earlier in clean:
                    pair = (earlier, label) if earlier < label else (label, earlier)
                    result = compare_outputs(children[pair[0]].out_dir, children[pair[1]].out_dir)
                    agreement[pair] = result.score
                    logger.info("Consensus: %s vs %s agreement %.3f", *pair, result.score)
                    if result.agree and decided is None:
                        decided = pair[0]
                        self.stop_event.set()  # the others stop at their next attempt
                clean.append(label)

        clean.sort()
        if decided is None and len(clean) >= 3:
            for i, x in enumerate(clean):  # candidates that finished after the others
                for y in clean[i + 1 :]:
                    if (x, y) not in agreement:
                        agreement[(x, y)] = compare_outputs(
                            children[x].out_dir, children[y].out_dir
                        ).score

            def support(label: str) -> float:
                return sum(v for pair, v in agreement.items() if label in pair)

            decided = max(clean, key=support)
            logger.info(
                "Consensus medoid: %s (support %s)",
                decided,
                {c: round(support(c), 3) for c in clean},
            )
        elif decided is None and len(clean) == 2:
            x, y = clean
            pick = self._judge(
                (x, children[x].accepted_code, children[x].out_dir),
                (y, children[y].accepted_code, children[y].out_dir),
                compare_outputs(children[x].out_dir, children[y].out_dir).diffs,
            )
            decided = pick[0]
        elif decided is None and clean:
            decided = clean[0]
        if decided is None:
            logger.error("Consensus: no candidate produced a clean run")
            return False
        restore_outputs(children[decided].out_dir, self.out_dir)
        self._save_solution(children[decided].accepted_code)
        logger.info(
            "Consensus: submitted candidate %s of clean %s (model_requests=%s)",
            decided,
            clean,
            getattr(self.client, "request_count", "unknown"),
        )
        return True
