import pandas as pd
import pytest

from biodb import cellxgene


def test_failed_cell_type_does_not_complete_aggregate_and_successes_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(cellxgene, "_cellguide_snapshot", lambda: "snapshot")
    monkeypatch.setattr(
        cellxgene,
        "list_cellguide_cell_types",
        lambda: pd.DataFrame({"cell_ontology_id": ["CL:1", "CL:2"]}),
    )
    calls = []

    def fetch(cl_id, **kwargs):
        calls.append(cl_id)
        if cl_id == "CL:2" and calls.count("CL:2") == 1:
            raise ConnectionError("failed")
        return pd.DataFrame({"cell_ontology_id": [cl_id], "gene_symbol": ["GENE"]})

    monkeypatch.setattr(cellxgene, "cellguide_markers", fetch)
    with pytest.raises(RuntimeError, match="incomplete"):
        cellxgene.get_all_cellguide_markers(cache_dir=tmp_path, max_workers=1, progress=False)
    assert not list(tmp_path.glob("cellguide_markers*.parquet"))
    result = cellxgene.get_all_cellguide_markers(cache_dir=tmp_path, max_workers=1, progress=False)
    assert len(result) == 2
    assert calls.count("CL:1") == 1
    assert calls.count("CL:2") == 2
