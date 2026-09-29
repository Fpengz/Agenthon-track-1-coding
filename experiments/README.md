# Experiment results management

Local evaluation runs of the agent are practice feedback for Track 1, not a prediction of the Final
rank. This folder keeps them comparable and reproducible.

## Layout

```
experiments/
  README.md          # this standard (tracked)
  registry.jsonl     # one line per finished run: provenance, metrics, every unit's status (tracked)
  runs/<run_id>/     # raw artifacts (gitignored, prunable)
    manifest.json    # written at start: what produced the run
    summary.json     # all per-unit results + competition metrics
    results.jsonl    # per-unit results as they finish (survives an interrupted run)
    <unit>/          # agent.log, checker.log, output/, meta/{agent_status.json,transcripts/}
```

Each unit's staged `input/` copy is deleted as soon as the unit finishes (`--keep-inputs` keeps
it): it is reproducible from `units/` and was most of a run's disk use.

## Rules

1. **Every run goes through `solve-units` with a name and a note.** The run id is
   `YYYYMMDD-HHMM-<name>`; the note says what the run tests.

   ```bash
   docker build -t agenthon-agent:latest -f Dockerfile.agent .
   uv run python -m agent.main solve-units -j 8 --agent-image agenthon-agent:latest \
     --name notes-on --note "domain notes in the prompt" --baseline 20260929-0816-review-on
   ```

2. **Commit before a run you want to keep.** The manifest records the git commit and a `dirty`
   flag; a dirty run cannot be reproduced from the commit and is marked `*` in `runs list`.
   It also records the Docker image *ID* (tags get rebuilt), the checker image, the served model,
   every behaviour setting (`HOUSE_REASONING`, `AGENT_MAX_REVIEWS`, `AGENT_TIME_BUDGET_SEC`,
   `HOUSE_CONTEXT_TOKENS`, `HOUSE_REQUEST_TIMEOUT`, `AGENT_DOMAIN_NOTES`, `AGENT_CANDIDATES`), `-j`,
   and the unit
   list. Each run also keeps its own `run.log`.

3. **Change one thing per comparison**, and name the run it is compared with (`--baseline`).

4. **Compare with at least 2 runs per variant.** The same code flips units between runs: across
   three runs of similar code, 18 units passed at least once but far fewer passed every time.
   `runs compare` reports pass@1 on the units all runs share, per-unit gains and losses, and the
   passed-in-any vs passed-in-every gap that measures this noise.

5. **Report the official metric**: mean pass@1 over a fixed denominator (the planned units;
   timeouts, crashes and unreached units count as failures). Gate estimates, labels, check pass
   fractions, requests and time are diagnostics.

6. **Keep load comparable.** Per-request speed is flat up to about 8 concurrent jobs on the local
   vLLM and ~23% slower at 12; the per-unit time budget counts that time. Compare runs made at the
   same `-j`, or raise `AGENT_TIME_BUDGET_SEC` in proportion (and record it; it is in the manifest).

7. **Prune raw artifacts, never registry lines.** `registry.jsonl` is append-only; `runs prune`
   deletes a run's folder and keeps its record. Keep the raw folder of any run a few-shot library
   is built from (`build-examples` reads its `meta/transcripts/solution.py` files).

8. **Nothing here may leak answers or secrets.** Registry lines hold statuses and metrics, never
   outputs, scripts or tokens (the manifest records the model endpoint and name, not
   `MODEL_TOKEN`). The few-shot library (`agent/examples/library.jsonl`) contains solutions to
   public units: it is gitignored and must not be pushed to a public repository.

## Commands

```bash
uv run python -m agent.main runs list [-n 10]            # recorded runs
uv run python -m agent.main runs compare BASE OTHER ...  # shared-unit pass@1 and flips
uv run python -m agent.main runs prune RUN_ID ...        # delete raw artifacts, keep the record
```

`--no-register` skips the registry for throwaway smoke runs; `--run-dir` overrides the location.
