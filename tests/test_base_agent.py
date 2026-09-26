"""Tests for the base agent implementation."""

import pathlib
import tempfile
import pandas as pd
from agent.executor import extract_python_code, run_code
from agent.loop import AgentSolver


def test_extract_python_code():
    markdown = """Here is the solution:
```python
import numpy as np
print("hello")
```
Hope this helps!"""
    extracted = extract_python_code(markdown)
    assert extracted == 'import numpy as np\nprint("hello")'


def test_executor_run():
    with tempfile.TemporaryDirectory() as tmp_dir:
        work = pathlib.Path(tmp_dir)
        code = 'print("hello executor")'
        result = run_code(code, work, work)
        assert result.success
        assert "hello executor" in result.stdout


class DummyMockClient:
    def __init__(self):
        self.request_count = 0

    def chat(self, messages, **kwargs):
        self.request_count += 1
        return '''```python
import pandas as pd
import pathlib
import os

out = pathlib.Path(os.environ.get("OUTPUT_DIR", "."))
df = pd.DataFrame({"option_id": [1], "price": [10.0]})
df.to_parquet(out / "results.parquet")
```'''


def test_agent_solver_mock():
    with tempfile.TemporaryDirectory() as tmp_out:
        task_dir = pathlib.Path("units/t1-EXAMPLE-bs-greeks-pde")
        out_dir = pathlib.Path(tmp_out)
        solver = AgentSolver(
            task_dir=task_dir,
            out_dir=out_dir,
            client=DummyMockClient(),
        )
        success = solver.run()
        assert success
        assert (out_dir / "results.parquet").exists()
