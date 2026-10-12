"""Tests for the program synthesizer module."""

import pytest
from solon.config import load_dataset_config
from solon.pipeline.program_synthesizer import (
    extract_function, validate_program, load_program, run_program,
)


@pytest.fixture
def heloc_config(full_heloc_yaml_path):
    return load_dataset_config(full_heloc_yaml_path)


class TestExtractFunction:
    def test_extracts_from_code_block(self):
        response = "Here is the code:\n```python\ndef underwrite(app):\n    return {'decision': 'APPROVE'}\n```\nDone."
        code = extract_function(response)
        assert "def underwrite" in code

    def test_extracts_bare_function(self):
        response = "def underwrite(app):\n    return {'decision': 'APPROVE'}"
        code = extract_function(response)
        assert "def underwrite" in code

    def test_extracts_with_imports(self):
        response = "```python\nimport numpy as np\n\ndef underwrite(app):\n    return {'decision': 'DENY'}\n```"
        code = extract_function(response)
        assert "import numpy" in code
        assert "def underwrite" in code


class TestValidateProgram:
    def test_valid_program(self, heloc_config, sample_program_code):
        valid, msg = validate_program(sample_program_code, heloc_config)
        assert valid is True
        assert msg == "OK"

    def test_rejects_target_leakage(self, heloc_config):
        code = "def underwrite(app):\n    if app['RiskPerformance'] == 1: return {}"
        valid, msg = validate_program(code, heloc_config)
        assert valid is False
        assert "RiskPerformance" in msg

    def test_rejects_dangerous_ops(self, heloc_config):
        code = "import os\ndef underwrite(app):\n    os.system('rm -rf /')"
        valid, msg = validate_program(code, heloc_config)
        assert valid is False
        assert "Dangerous" in msg

    def test_rejects_syntax_errors(self, heloc_config):
        code = "def underwrite(app incomplete"
        valid, msg = validate_program(code, heloc_config)
        assert valid is False
        assert "Syntax" in msg


class TestLoadAndRunProgram:
    def test_load_and_run(self, heloc_config, sample_program_code, tiny_dataset):
        fn = load_program(sample_program_code)
        results = run_program(fn, tiny_dataset, heloc_config)
        assert len(results) == len(tiny_dataset)
        for r in results:
            assert r["decision"] in ("APPROVE", "DENY", "ERROR")
            assert r["error"] is None or isinstance(r["error"], str)

    def test_target_column_stripped(self, heloc_config, sample_program_code, tiny_dataset):
        """The program should never see the target column."""
        fn = load_program(sample_program_code)
        results = run_program(fn, tiny_dataset, heloc_config)
        # If we get here without errors, the target column was stripped
        assert all(r["error"] is None for r in results)
