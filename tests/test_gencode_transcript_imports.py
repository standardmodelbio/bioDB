"""Core GENCODE APIs remain importable without transcript acquisition helpers."""

import os
import subprocess
import sys


def test_core_gencode_import_does_not_load_optional_transcript_io():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc, sys
class BlockTranscriptIO(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'biodb.transcript_io':
            raise ImportError('transcript acquisition not installed')
sys.meta_path.insert(0, BlockTranscriptIO())
from biodb import gencode
assert callable(gencode.get_genes)
assert callable(gencode.fetch_mane_select)
""",
        ],
        capture_output=True,
        text=True,
        env=os.environ,
    )
    assert result.returncode == 0, result.stderr
