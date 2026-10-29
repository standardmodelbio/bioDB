"""Core acquisition must import without optional ontology/ML dependencies."""

import subprocess
import sys


def test_core_import_without_optional_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "for name in ('networkx', 'torch', 'matplotlib', 'datashader', 'owlready2'):\n"
            "    sys.modules[name] = None\n"
            "from biodb import gwas_catalog, opentargets, europepmc\n"
            "assert callable(gwas_catalog.ingest_sumstats)\n"
            "assert callable(opentargets.get_traits)\n"
            "assert callable(europepmc.fetch_metadata)\n",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
