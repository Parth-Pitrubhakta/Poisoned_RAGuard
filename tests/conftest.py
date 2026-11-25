import ast
import importlib.util
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream"
sys.path.insert(0, str(ROOT / "src"))


def _extract_functions(path: Path, names: list[str]) -> dict:
    """Exec only the named top-level functions from an upstream file (avoids its beir/fastchat imports)."""
    tree = ast.parse(path.read_text())
    src = "\n\n".join(ast.get_source_segment(path.read_text(), n) for n in tree.body
                      if isinstance(n, ast.FunctionDef) and n.name in names)
    ns = {"np": np}
    exec(compile(textwrap.dedent(src), str(path), "exec"), ns)
    return {n: ns[n] for n in names}


@pytest.fixture(scope="session")
def upstream_utils():
    return _extract_functions(UPSTREAM / "src" / "utils.py", ["clean_str", "f1_score"])


@pytest.fixture(scope="session")
def upstream_prompts():
    spec = importlib.util.spec_from_file_location("upstream_prompts", UPSTREAM / "src" / "prompts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # prompts.py has no imports
    return mod
