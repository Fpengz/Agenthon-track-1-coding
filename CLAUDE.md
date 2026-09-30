# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This is the single source of agent guidance for the repo; `AGENTS.md` only points here.

## Purpose

Use this checkout to build, practice, and package your team's Track 1 coding agent. This is the public practice repository; it does not contain the sealed Final evaluation tasks, reference answers, canary registry, or final scorer. Treat public-unit results as practice feedback, not a prediction of the Final rank.

Keep competition work in the participant agent and its image. During agent work, preserve the public unit inputs, checks, manifests, and scoring code. Do not add oracle material, private evaluation data, answer keys, or copied expected outputs. Never try to access or infer held-out material through private repositories, leaderboard probing, or scoring behavior. If explicitly asked to contribute a public task or maintain the benchmark, follow the task-author or maintainer documents listed below.

## Read the right contract

- [README.md](README.md) is the participant workflow, including the local agent-then-checker flow.
- [SUBMISSION_CLI.md](SUBMISSION_CLI.md) is the detailed interface, model-access, scoring, and submission contract.
- [baselines/README.md](baselines/README.md) documents House-model access and packaging. Track 1 has an interface exemplar, not an official baseline agent.
- [docs/CONCEPTS.md](docs/CONCEPTS.md) explains the gates and scoring terms. Read [docs/CATEGORIES.md](docs/CATEGORIES.md) for the domain invariants relevant to a task.
- [docs/QFBENCH-HERITAGE.md](docs/QFBENCH-HERITAGE.md) explains the runner and output conventions inherited from QFBench.
- For task authoring, read [docs/AUTHORING-GUIDE.md](docs/AUTHORING-GUIDE.md). For organizer dataset builds, read [docs/DEVELOPMENT-DATASET-BUILDER.md](docs/DEVELOPMENT-DATASET-BUILDER.md). The trusted-checker documents apply when explicitly working on grader tooling.

The official [competition page](https://www.agenthon.net/compete/), [Rules](https://www.agenthon.net/rules/), and [submission guide](https://www.agenthon.net/guides/submission-format/) govern participation. Guides are explanatory; the Rules and current track instructions control if details conflict. Check the signed-in account for current CodaBench access and upload status.

## Build and test the entrant

1. Improve the team's agent in the agent package and its container configuration. The Track 1 image must implement the exact command **solve --task-dir <path> --out <path>**. In the evaluation container, task input is mounted read-only at **/input** and deliverables go to **/app/output**.
2. For every task, read its own instruction and inspect its supplied input files. Follow that unit's exact output filenames, schema, units, data cutoff, and timeout from its card. Do not generalize the exemplar's filenames or formats to other tasks.
3. Keep all required code, packages, and permitted artifacts in the image. Evaluation has no general internet access and permits only the restricted House-model route for model calls. Do not depend on runtime package installs, external data fetches, vendor APIs, or vendor-side tools such as web search, retrieval, or code execution.
4. Use the House model substantively at run time to solve each task. Track 1 submissions use category **api**; model-free runs do not earn credit. The published rule requires a House-model solution for each counted pass from **2026-10-05 00:00 AoE**. A compliance-only call followed by a prepared answer does not qualify.
5. Respect the House allowance: at most **25 admitted requests per unit** and at most **4,000 output tokens per request**. Failed admitted requests and retries count. There is no separate total output-token pool per unit. Keep the limits; make prompts and repair loops efficient.
6. Follow the task's data cutoff for raw data and derived material, including features, caches, indexes, and generated data. Keep canary GUIDs and other task metadata out of deliverable files.
7. Team-authored solutions to public practice tasks may be included as examples for solving other tasks only, under the current Track 1 rules. Never use a stored answer to solve the same task it answers. General-purpose algorithms and libraries are allowed when licensed and disclosed as required.
8. Validate with a fresh output directory for each task. The local smoke checker checks existing output; it does not launch the agent. Follow [README step 6](README.md#6-run-your-agent-then-check-its-output) to run the agent and checker, require reward 1.0, and inspect pytest_report.json. Harbor pass@3 is an optional offline diagnostic, not the official leaderboard score.
9. Treat the public scorer as a validator, not as code to tune for a higher score. Report a demonstrable evaluation defect through the published organizer channel instead of changing unit checks or scoring behavior in the entrant image.

## Scoring and submission

Track 1's official metric is mean pass@1: one execution per task over a fixed denominator. A wrong, crashed, timed-out, missing, or unreached result remains a failure. Every counted pass must clear **g0_integrity**, **g1_schema**, **g2_cutoff_resource**, and **g3_domain_semantics**. Public Development units are practice; Final uses sealed held-out tasks. There is no official baseline score to beat.

Build the agent as a Linux/amd64 Docker image, pin the submitted image by its immutable digest, and declare the actual House-model details in the current submission descriptor. Use the shared toolkit's submission commands to create the team alias and package submission.zip with submission.json and the toolkit-generated team-claim.json. The image is fetched separately. Do not put a Team Key, password, registry credential, or private verification proof in source control or public messages. The ordinary public-image route requires an anonymous pull by digest; a confidential mirror needs prior organizer confirmation. Upload through the team's designated CodaBench account, linked to the registered Agenthon team.

As checked on 2026-09-29, the official schedule lists registration and Development closing on 2026-10-12 at 23:59 AoE, with the last Development runs required to start by 20:00 UTC that day. Track 1 Development allows one upload per team per day and 23 total uploads; held or cancelled uploads count, while local checks and packaging do not. Final + Verification is listed for 2026-10-13 through 2026-10-25 at 23:59 AoE, with one team submission per entered track and organizer verification in the same phase. Recheck the official site and signed-in account before planning or uploading; schedule and availability are controlled there. Teams have one to three registered members, and each person belongs to one team.

## External actions and working tree

Keep registration, accepting competition terms, Team Keys, and CodaBench uploads under the participant's control; perform them only on the participant's explicit request. Preserve existing user changes. Keep generated outputs, logs, model/cache data, credentials, and build artifacts out of source control; inspect git status before preparing a commit.

## Repository layout (big picture)

Two separate things share this repo:

- **`agent/`** — the team's Track 1 entrant (the thing we improve). Packaged by `Dockerfile.agent`
  (`FROM finance-bench-sandbox:latest`, entrypoint `python -m agent.main solve ...`).
- **Organizer benchmark material** — `units/` (86 public practice tasks), `qfbench2_track_coding/`
  (verifier/scorer, built on the `qfbench2-common` toolkit pinned at tag v2.4.3), `docker/`
  (shared sandbox base image), `templates/`, `scripts/`, `.github/` and most of `tests/`.
  Treat these as read-only validators; do not edit unit checks, manifests or scoring code.

Each unit is `units/<id>/` with `instruction.md` (what the agent sees), `card.toml` (authoritative
per-unit limits, e.g. `[agent].timeout_sec`, `data_cutoff`, canary GUID), `manifest.json`,
`environment/data/` (inputs) and `checks/` (`test.sh` + `test_outputs.py`, run offline after the
agent exits; they write `reward.json` / `pytest_report.json`).

## Agent architecture (`agent/`)

Generate → execute → self-repair loop, one House-model request per attempt:

- `main.py` — Typer CLI (`solve --task-dir --out`, plus the local-only `solve-units` batch
  runner); loads `.env` from the repo root for local runs. `solve` always exits 0 so the checker
  still records the attempt; its hidden `--status-file` reports success/request count to the batch
  runner (`batch.py`).
- `loop.py` — `AgentSolver`: reads `instruction.md`, discovers inputs (skipping `checks/` and
  `reference/`, which hold graded answers in a local checkout and are never mounted at
  evaluation), infers expected deliverable names (prefers `/output/<name>` mentions, charts and
  reports included), then loops up to `max_retries + 1` attempts. Each
  request is rebuilt as `[system, latest user prompt]` — history is **not** accumulated, because
  the repair prompt already carries the previous code + error and replaying the model's reasoning
  overflows its context. Stale expected deliverables are deleted before each execution and
  outputs must be newer than the execution start. Every turn (including repairs) resends the
  full task prompt, since there is no history. A script cut off mid-block at the 4,000-token cap
  is continued (up to `MAX_CONTINUATIONS` follow-ups, stitched by `join_continuation`; the cut
  part is first trimmed to its last complete logical line via `tokenize`, otherwise a cut inside
  a multi-line bracket makes the continuation restart it and leaves it unclosed); a
  response cut off before any code is handed back as "previous analysis".
- `consensus.py` + `_solve_with_consensus` (`loop.py`) — best-of-N, `AGENT_CANDIDATES` (default 1
  = off). Candidates run IN PARALLEL as child `AgentSolver`s sharing the client (request
  admission is atomic), budget, deadline and prompt parts; each writes to its own staging
  OUTPUT_DIR. A is the regular loop (greedy, with review); B, C... are sampled (0.7), unreviewed,
  <= 6 attempts. The first agreeing pair (>= 0.95) sets a stop event; otherwise 3+ clean
  candidates submit the medoid and 2 use a judge request. Why parallel: the budget, not time, is
  what goes unused (failing units used 7.7 of 25 requests; 115 of 136 ended "accepted but
  wrong"), and the sequential version cost 521 s per unit, over the roster allowance.
- `spec.py` — `AGENT_SPEC_CHECKS` (default off): parsed from the instruction (headings, column
  tables, bold-bullet paragraphs, `Columns:` sentences, JSON examples with `<float>` placeholders;
  quoted placeholder KEYS like `"<sector>"` are dynamic and never required): required columns /
  JSON key paths, numeric columns (Type cell) and numeric JSON values, exact "N rows". After a
  clean run, findings trigger up to 2 repair turns. Parses 64/86 units. Rule: every check must
  flag 0 checker-passed outputs (currently 0/95) before it is kept.
- `context.py` (always on) — canary hygiene and task context. 59/86 instructions carry their own
  canary GUID (HTML comment or `# ...-canary GUID` heading); a canary in a deliverable is a g2
  disqualification. Instructions are sanitised before prompting, and every clean run's
  deliverables are scanned: a leak is never snapshotted or accepted (repair turn instead). The
  prompt gets an allow-list of card facts (category, difficulty, data cutoff, time limit, compute).
- `preflight.py` (always on) — static checks before a script runs (no request): BLOCKING =
  certain failures (syntax/indentation errors; module-level imports, outside try blocks, of
  modules neither installed nor shipped as .py in TASK_DIR) -> repaired without running;
  ADVISORY = likely bugs that may sit on paths never executed (pyflakes undefined names, `/app/`
  unit-image paths, pandas-3-removed APIs) -> appended to a failed run's feedback. Validated
  inside the agent image: 0 blocking flags on 2,781 scripts that ran cleanly (undefined names
  were 1.7%, which is why they only advise). Needs pyflakes (pinned in `Dockerfile.agent`).
- `error_context.py` (always on) — `sitecustomize.py` written next to each script: on an
  uncaught exception it prints an `[agent]` block (failure type, failing line, DataFrame
  columns/shape/index, dict keys, array shapes of the variables in scope). 28% of failures used
  to repeat the previous error. Repair prompts start with `Failure type: ...`.
- Experimental switches, all OFF (A/B-tested, not adopted; code kept): `AGENT_SPEC_CHECKS`
  (0.157-0.180 vs control), `AGENT_EXAMPLES` (few-shot library; 0.157), `AGENT_STRUCTURED` (one
  function per deliverable; 0.163), `AGENT_CANDIDATES` (consensus; same gain as low reasoning at
  12.7 requests / 521 s per unit, over the roster budget), `AGENT_DOMAIN_NOTES` (0.140 vs 0.157).
  Results per run are in `experiments/registry.jsonl`.
- Methods that spend the unused request budget (all default off; A/B pending):
  `HOUSE_REASONING=hybrid` (low effort for generation, off for review/repair/continuation; the
  not-honoured detector only judges generation replies), `AGENT_VERIFY=1` (model-written
  verifier executed on a COPY of the outputs; explicit `FAIL:` lines drive a repair, and the
  model may return its script unchanged to keep it; <= 2 rounds), `AGENT_PLAN=1` (one request
  extracts a requirements checklist into every prompt), `AGENT_EXPLORE=K` (tool use: up to K
  ```explore snippets run read-only with output fed back before the final script),
  `AGENT_SKILLS=1` (advertise `agent_skills`, vetted helpers from `skills.py` copied next to
  every script: strict JSON writer, table reader, annualisation, Sharpe, drawdown).
- Adaptive evidence budget (`AGENT_ADAPTIVE=1`, default off): the client tracks request latency
  (p75 of the last 8; prior `HOUSE_LATENCY_PRIOR_SEC`, default 45), and `_can_afford(n)`
  replaces the fixed review/verifier caps (up to 4 reviews / 3 verifier rounds while
  "requests left and time left > n x latency + run time + margin"). Local latency (~22 s single
  stream, ~43 s under A/B load) overstates the House route's, so A/B time figures are upper
  bounds; the real latency is in any Development upload's logs (`completed in X s`).
- `review.py` — self-verification. Exit 0 + present files says nothing about correctness and the
  checker is sealed, so a clean run is not accepted immediately: the loop snapshots the outputs,
  computes mechanical findings (NaN/inf, empty tables, JSON nulls) and sends a review prompt
  (spec + script + output previews). `VERDICT: PASS` accepts; `VERDICT: FAIL` + a corrected script
  goes through the normal run/repair path; a FAIL cut off before its script becomes a repair turn
  fed the review's findings. At most `MAX_REVIEWS` per unit and none after `REVIEW_DEADLINE_SEC`;
  once a clean run exists, rewrites get a tighter execution timeout (10x its runtime, 60-300 s);
  if a "fix" never runs cleanly the snapshot is restored. Repair prompts flag an error that
  repeats the previous attempt's so the model changes approach instead of re-patching. If the
  model copies the failing script back verbatim (near-deterministic even when sampling), it is
  not re-run: the next request asks for a fresh script from the task prompt without the old code.
  `AGENT_MAX_REVIEWS` (default 2; 0 disables) is forwarded into agent containers for A/B runs.
- Edit-based repairs: for scripts of `EDIT_MODE_MIN_LINES`+ lines, repair and review prompts ask
  for SEARCH/REPLACE blocks (`apply_edits` in `executor.py`; exact match, then a unique
  trailing-whitespace-tolerant line match) instead of a rewrite. Full rewrites of ~300-line
  scripts overran the 4,000-token cap on every repair (2+ requests each) and the model tended to
  copy the old script back; this was the main cause of units ending with no output at all.
- `AGENT_TRANSCRIPT_DIR` (debug; `solve-units` sets it to `<unit>/meta/transcripts`, outside the
  deliverables) saves every request/response pair. Unset at evaluation.
- `knowledge.py` — offline domain notes (OFF by default; `AGENT_DOMAIN_NOTES=1` enables them —
  a 2x2 A/B showed no gain: 0.140 on vs 0.157 off): the "Financial invariants" and "Common mistakes"
  subsections of `docs/CATEGORIES.md` for the 1-2 most relevant categories (TF-IDF over the
  instruction + card category/tags; card categories are free-form). `Dockerfile.agent` copies
  the doc to `agent/data/CATEGORIES.md`; locally it is read from `docs/`.
- `examples.py` — few-shot reference example from `agent/examples/library.jsonl` (rule 8: the
  team's own solutions to public units, used for OTHER units only). Excluded: same unit id, same
  instruction hash, or instruction Jaccard >= 0.5 (template re-issues such as
  `t1-momentum-backtest`/`t1-sma-crossover-spy`). GUIDs are stripped. Build/extend the library
  with `uv run python -m agent.main build-examples experiments/runs/<run_id> [...]`, which harvests
  checker-passed units' accepted scripts (saved as `meta/transcripts/solution.py`).
- `experiments.py` — run ids, manifests, the registry, `runs list/compare/prune` (host tooling;
  standards in `experiments/README.md`: commit before runs you keep, one change per comparison,
  >= 2 runs per variant because units flip between identical runs, same `-j` for comparisons).
- `inputs.py` — bounded previews (tables also get facts before the sample rows: dtypes, nulls,
  date ranges, first-column uniqueness, duplicate rows -- a data inspector without request
  round-trips) of each input file (CSV head + row count, Parquet schema +
  rows, JSON key skeleton, xlsx sheets, ...). Without them the model guesses column names and
  JSON keys and repeats the same `KeyError` on every repair.
- `container_path_map` (in `loop.py`): 40 instructions cite image paths like
  `/app/data/stock_data.parquet`, created by the unit's `environment/Dockerfile` COPY lines —
  sometimes renamed (from `environment/data/stock_chars.pqt`). A submission never runs in the
  unit image, so the prompt lists each cited path -> its TASK_DIR file.
- `executor.py` — `extract_python_code` strips inline reasoning up to the last `</think>`, accepts
  only *closed* fenced blocks (prefers the last one that compiles) and returns `""` otherwise —
  prose is never executed. `run_code` runs the script in a scratch temp dir (never in the output
  dir) with env vars `TASK_DIR` and `OUTPUT_DIR`.
- `client.py` — `HouseModelClient` (OpenAI client at `$MODEL_ENDPOINT/v1`, bearer `$MODEL_TOKEN`,
  model `$MODEL_NAME`); enforces 25 requests/unit and ≤4,000 output tokens; `chat()` returns
  `ChatResult(content, finish_reason)` so truncation (`finish_reason == "length"`) is detectable.
- `prompts.py` — system/initial/repair/continuation/no-code prompts. The system prompt reports the
  running interpreter's library versions (the image's, at evaluation: pandas 3). Generated scripts must map the
  instruction's `/input` → `TASK_DIR` and `/output` → `OUTPUT_DIR` (paths differ between local runs
  and the container).

### House model behaviour (observed locally)

- Served model is NVIDIA Nemotron-3 Super via vLLM with a **32,768-token context**. Reasoning is
  emitted inline in `content`, terminated by `</think>` (no opening tag, no reasoning parser).
- With full reasoning, responses routinely spend the whole 4,000-token cap on reasoning and emit
  no code. `HOUSE_REASONING=low|off|on` (default **low**) maps to `chat_template_kwargs`
  `{"low_effort": true}` / `{"enable_thinking": false}` / server default. `low` won two parallel
  A/Bs (4 runs 16-19/86 vs 15-17 without it; no-output failures halved; ~20% fewer requests). If a
  reply shows low effort is not honoured (> 4,000 reasoning chars, or cut off before any code),
  the client falls back to `off` for the rest of the unit.
  The hub's `docs/HOUSE-MODEL.md` documents `enable_thinking` as the thinking control and says the
  route forwards `low_effort` and `reasoning_budget` untested; `reasoning_budget` had no effect on
  the local vLLM.
- Context window: the House docs state no serving window (tokenizer metadata 262,144 "does not
  establish an allowed request size"); local vLLM serves 32,768. An over-long request is rejected
  upstream after admission, so it likely still costs one of the 25 requests. `fit_to_context`
  (`loop.py`) estimates tokens at 2.5 chars/token (lowest measured ratio 2.53, median 3.24) and,
  if a request would not fit `HOUSE_CONTEXT_TOKENS` (default 32768), swaps in a leaner task prompt
  (dropping reference example -> domain notes -> input previews, kept for later turns), then
  shrinks the output cap, and stops only if under 1,024 output tokens would remain.
- The OpenAI SDK is built with `max_retries=0`: SDK retries are admitted House requests that would
  bypass the 25-request counter.

## Commands

```bash
uv sync --group dev                      # install (Python >= 3.13)
uv run ruff check agent                  # lint
uv run ruff format --check agent         # format check
uv run ty check agent                    # type check
uv run python -m pytest -q -p no:cacheprovider tests/test_base_agent.py          # agent tests
uv run python -m pytest -q tests/test_base_agent.py::test_extract_python_code    # single test

# Run the agent locally against a unit (needs MODEL_ENDPOINT/MODEL_NAME, e.g. from .env).
# Use a fresh --out directory every run.
uv run agent/main.py solve --task-dir units/t1-EXAMPLE-bs-greeks-pde --out ./tmp/<fresh-dir>
# (tmp/ is gitignored scratch for ad-hoc runs; real experiments follow experiments/README.md.)

# Run the agent over many units (all by default; name globs select), N in parallel.
# Stages each unit without checks/ and reference/ (as evaluation mounts it), runs solve in a
# subprocess under the card's timeout, then runs the unit checker in the sandbox image if it
# exists (as your user; inputs also mounted at their /app COPY targets and /tests/reference_data,
# like the official scorer). Every run follows experiments/README.md: run id
# <YYYYMMDD-HHMM>-<name>, manifest.json (git commit/dirty, image IDs, model, settings) and raw
# artifacts under experiments/runs/<run_id>/ (gitignored; staged input copies deleted per unit),
# plus one line in the tracked, append-only experiments/registry.jsonl when it finishes.
uv run python -m agent.main solve-units -j 8 --agent-image agenthon-agent:latest \
  --name notes-off --note "what this run tests" --baseline 20260929-0816-review-on
uv run python -m agent.main solve-units 't1-EXAMPLE-*' --no-check --timeout 600 --no-register
uv run python -m agent.main runs list                        # recorded runs
uv run python -m agent.main runs compare BASE OTHER [...]    # shared-unit pass@1 + per-unit flips
uv run python -m agent.main runs prune RUN_ID                # drop raw artifacts, keep the record
# Faithful mode: run solve inside the agent image (as evaluation does). Prefer it for scores:
# the local venv lacks sandbox libraries generated scripts use (statsmodels, arch, matplotlib,
# plotly, openpyxl, ta-lib, ...), so host-mode failures can be environment artifacts.
docker build -t agenthon-agent:latest -f Dockerfile.agent .
# Each run ends with competition-style metrics (agent/metrics.py; also summary.json "metrics"):
# official mean pass@1 over the fixed denominator, g0-g3 gate estimates, official failure labels
# (g3 label via the organizer's own classifier), pass@1 by difficulty/category, and diagnostics
# (checks passed, House requests incl. rule-9 zero-request units, time vs the 12 h allowance).
# Rebuilding an image tag while a run uses it switches later units to the new code: use a new tag.

# Official local check (README step 6): build the sandbox once, then run agent + checker in
# Docker so the unit is mounted at /input and output at both /app/output and /output.
docker build -t finance-bench-sandbox:latest -f docker/sandbox.Dockerfile .
docker build -t <agent-image> -f Dockerfile.agent .
```

`Dockerfile.agent` pins the client libraries to the `uv.lock` versions; keep them in sync. The
earlier `openai==1.40.0` pin crashed on client construction against the base image's httpx 0.28.

Checker tests hardcode container paths (`/input/...`, and for 40 units `/app/data/...` or
`/tests/reference_data`), so running `checks/test_outputs.py` on the host fails with
`FileNotFoundError`. The bare README step 6 command mounts only `/input`, so those 40 units fail
there regardless of output; `solve-units` adds the scorer's input mounts.
Require `reward: 1.0` in `reward.json` and inspect `pytest_report.json` — `test.sh` exits 0 even
when checks fail.

Organizer CI (`.github/workflows/ci.yml`) lints only `qfbench2_track_coding` and runs the
scorer/firewall tests; it does not exercise `agent/`.
