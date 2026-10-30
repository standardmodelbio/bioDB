"""Publication checks for saved notebook execution and offline validation."""

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import nbformat

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "validate_notebooks.py"
SPEC = importlib.util.spec_from_file_location("validate_notebooks", SCRIPT)
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class TutorialValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "example.ipynb"

    def write(self, *, count=1, tags=(), source="print('ok')", outputs=None):
        cell = nbformat.v4.new_code_cell(
            source,
            execution_count=count,
            metadata={"tags": list(tags)},
            outputs=outputs or [],
        )
        notebook = nbformat.v4.new_notebook(cells=[cell], metadata={"biodb": {"offline": True}})
        nbformat.write(notebook, self.path)
        return notebook

    def test_executed_cell_is_publishable(self):
        self.write(outputs=[nbformat.v4.new_output("stream", name="stdout", text="ok\n")])
        self.assertEqual(VALIDATOR.validate(self.path).cells[0].execution_count, 1)

    def test_unexecuted_saved_cell_is_rejected(self):
        self.write(count=None)
        with self.assertRaisesRegex(ValueError, "not been executed"):
            VALIDATOR.validate(self.path)

    def test_skip_tag_cannot_hide_a_code_cell(self):
        self.write(tags=["skip-execution"], source="raise RuntimeError('must execute')")
        with self.assertRaisesRegex(ValueError, "skip-execution"):
            VALIDATOR.validate(self.path)

    def test_saved_error_is_rejected(self):
        self.write(outputs=[nbformat.v4.new_output("error", ename="ValueError", evalue="bad")])
        with self.assertRaisesRegex(ValueError, "saved ValueError"):
            VALIDATOR.validate(self.path)

    def test_invalid_code_is_rejected(self):
        self.write(source="if invalid syntax")
        with self.assertRaises(SyntaxError):
            VALIDATOR.validate(self.path)

    def test_offline_opt_in_is_required_for_execution(self):
        notebook = self.write()
        notebook.metadata.biodb.offline = False
        with self.assertRaisesRegex(ValueError, "offline = true"):
            VALIDATOR.execute(self.path, notebook, 10)

    def test_network_guard_blocks_before_connecting(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                VALIDATOR.NETWORK_GUARD + "\nsocket.create_connection(('example.com', 443))",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Tutorial execution is offline", result.stderr)


if __name__ == "__main__":
    unittest.main()
