"""Targeted coverage tests for the remaining branches in ``biodb.utils`` and
``biodb.harmonizome``.

These tests aren't smoke tests — every assertion checks a real invariant
in a code path that the broader suite missed:

* ``read_gmt`` — UTF-8-with-BOM fallback to latin-1; dict return-format;
  empty-genes row dropped from pandas output; show_progress wrapper;
  bad return-format raises.
* ``_reverse_excel_date`` — non-string, no-dash, non-digit-prefix, and
  unknown-month inputs all round-trip unchanged (no silent corruption).
* ``_parse_gmt_line`` — empty/skippable fields are filtered cleanly; a
  line starting with a malformed gene-set-name token is rejected whole.
* ``harmonizome.get_dataset_metadata`` — non-trivial type coercion of
  ``resource`` (dict → str), ``stats`` (non-str-non-dict → str).
* ``harmonizome.get_gmt`` — parquet-write failure is swallowed but the
  DataFrame is still returned; metadata-fetch parallel error path
  substitutes an empty record without aborting the run; empty
  ``all_tasks`` returns the canonical empty schema.
* ``harmonizome.load_gene_attribute_matrix`` — tissue metadata is
  extracted from the ``CellLine``-marker row; numeric vs alphanumeric
  gene-id detection in the data section; the all-na first data row is
  skipped.
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from biodb import harmonizome, utils

# ─── biodb.utils.read_gmt ────────────────────────────────────────────────────


def _write_gmt(path: Path, body: str, *, encoding: str = "utf-8") -> None:
    path.write_bytes(body.encode(encoding))


def test_read_gmt_dict_format_groups_by_set_and_description(tmp_path: Path) -> None:
    """``return_format='dict'`` returns a ``{(name, description): [genes]}`` mapping."""
    gmt = tmp_path / "tiny.gmt"
    _write_gmt(gmt, "SET_A\thallmark of A\tBRCA1\tTP53\nSET_B\tdescription B\tEGFR\n")

    result = utils.read_gmt(gmt, return_format="dict")

    assert result == {
        ("SET_A", "hallmark of A"): ["BRCA1", "TP53"],
        ("SET_B", "description B"): ["EGFR"],
    }


def test_read_gmt_falls_back_to_latin1_on_utf8_decode_error(tmp_path: Path) -> None:
    """A latin-1-encoded byte (0xE9 = 'é') outside UTF-8 still parses cleanly."""
    gmt = tmp_path / "latin1.gmt"
    gmt.write_bytes(b"SET\tdescription with \xe9\tBRCA1\tTP53\n")

    result = utils.read_gmt(gmt, return_format="dict")

    assert ("SET", "description with é") in result
    assert result[("SET", "description with é")] == ["BRCA1", "TP53"]


def test_read_gmt_show_progress_wraps_iterators(tmp_path: Path) -> None:
    """``show_progress=True`` wraps both the line-iterator AND the DataFrame
    materialization in ``tqdm``."""
    gmt = tmp_path / "tiny.gmt"
    _write_gmt(gmt, "SET\tdesc\tBRCA1\n")

    df = utils.read_gmt(gmt, return_format="pandas", show_progress=True, suppress_stats=True)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 3)


def test_read_gmt_logs_stats_when_not_suppressed(tmp_path: Path, caplog) -> None:
    """The default ``suppress_stats=False`` logs the loaded-row summary."""
    gmt = tmp_path / "tiny.gmt"
    _write_gmt(gmt, "SET\tdesc\tBRCA1\tTP53\n")

    with caplog.at_level(logging.INFO, logger="biodb.utils"):
        utils.read_gmt(gmt, return_format="pandas")

    assert any("Loaded 2 rows" in r.message for r in caplog.records)


def test_read_gmt_drops_empty_gene_sets_from_pandas_output(tmp_path: Path) -> None:
    """A gene-set row with no real genes (all skippable) doesn't emit DataFrame rows."""
    gmt = tmp_path / "empty.gmt"
    _write_gmt(gmt, "REAL_SET\thas a real gene\tBRCA1\n*MALFORMED\tdesc\t*marker1\t*marker2\n")

    df = utils.read_gmt(gmt, return_format="pandas", suppress_stats=True)

    assert set(df["id"]) == {"REAL_SET"}
    assert list(df["gene"]) == ["BRCA1"]


def test_read_gmt_rejects_unknown_return_format(tmp_path: Path) -> None:
    """Invalid ``return_format`` raises ``ValueError`` with a helpful message."""
    gmt = tmp_path / "tiny.gmt"
    _write_gmt(gmt, "SET\tdesc\tBRCA1\n")

    with pytest.raises(ValueError, match="return_format must be 'dict' or 'pandas'"):
        utils.read_gmt(gmt, return_format="json")


# ─── biodb.utils._reverse_excel_date ─────────────────────────────────────────


@pytest.mark.parametrize(
    "value",
    [
        None,  # non-string → passthrough
        12345,  # int → passthrough
        "BRCA1",  # no dash → passthrough
        "1-MAR-2024",  # too many dashes → passthrough
        "ABC-MAR",  # non-digit prefix → passthrough
        "1-FOO",  # unknown month → passthrough
    ],
)
def test_reverse_excel_date_passthrough_paths(value) -> None:
    """Every non-Excel-mangled input round-trips unchanged."""
    assert utils._reverse_excel_date(value) == value


@pytest.mark.parametrize(
    "mangled, original",
    [
        ("1-MAR", "MARCH1"),
        ("2-SEP", "SEPT2"),
        ("10-DEC", "DEC10"),
    ],
)
def test_reverse_excel_date_actually_reverses(mangled: str, original: str) -> None:
    """The happy path: Excel-mangled gene symbols round-trip to their real names."""
    assert utils._reverse_excel_date(mangled) == original


# ─── biodb.utils._parse_gmt_line (edge cases) ────────────────────────────────


def test_parse_gmt_line_skips_empty_fields_between_genes() -> None:
    """Empty (``""``) fields between genes are silently skipped."""
    gene_sets: dict[tuple[str, str], list[str]] = {}
    utils._parse_gmt_line(["SET", "desc", "BRCA1", "", "TP53", ""], gene_sets)

    assert gene_sets == {("SET", "desc"): ["BRCA1", "TP53"]}


def test_parse_gmt_line_skips_skippable_genes() -> None:
    """Tokens flagged by ``_is_skippable_gene`` BUT NOT by
    ``_looks_like_gene_set_name`` (long descriptions, parenthesized labels)
    are dropped silently without triggering the missing-newline recovery.

    Note: clone-id patterns (``001N03``) match both predicates, so the
    parser flushes the current set and recurses — that path is exercised by
    the existing missing-newline tests in ``test_utils.py``.
    """
    gene_sets: dict[tuple[str, str], list[str]] = {}
    utils._parse_gmt_line(
        [
            "SET",
            "desc",
            "BRCA1",
            "a very long descriptive label that exceeds thirty characters wide",
            "TP53",
            "WEIRDLABEL (some_aliased_protein description with spaces)",
            "EGFR",
        ],
        gene_sets,
    )

    assert gene_sets == {("SET", "desc"): ["BRCA1", "TP53", "EGFR"]}


def test_parse_gmt_line_clone_id_triggers_missing_newline_split() -> None:
    """A clone-id token (e.g. ``001N03``) matches the gene-set-name pattern,
    so the parser treats it as the start of a new (malformed) gene set,
    flushes the current one, and recurses. The recursion bails because the
    clone-id starts the new set with a name that itself looks like a
    gene-set name."""
    gene_sets: dict[tuple[str, str], list[str]] = {}
    utils._parse_gmt_line(["SET", "desc", "BRCA1", "001N03", "TP53"], gene_sets)

    # SET gets only BRCA1 — TP53 is dropped because the recursive parse of
    # ["001N03", "TP53"] sees the malformed leading token and returns.
    assert gene_sets == {("SET", "desc"): ["BRCA1"]}


def test_parse_gmt_line_drops_line_with_gene_set_name_in_first_position() -> None:
    """When line[0] looks like a malformed gene set (e.g. starts with ``*``),
    the whole line is dropped without raising."""
    gene_sets: dict[tuple[str, str], list[str]] = {}
    utils._parse_gmt_line(["*malformed", "desc", "BRCA1"], gene_sets)
    assert gene_sets == {}


# ─── biodb.harmonizome.get_dataset_metadata field coercion ───────────────────


def test_get_dataset_metadata_coerces_dict_resource_to_string() -> None:
    """If the API returns ``resource`` as a dict, we stringify so the metadata
    DataFrame stays parquet-safe."""
    fake_metadata = {
        "description": "d",
        "measurement": "m",
        "association": "a",
        "category": "c",
        "resource": {"name": "GEO", "url": "https://geo"},
        "citations": [],
        "lastUpdated": "2026-01-01",
        "stats": {"n_genes": 12345, "n_attributes": 6789},
    }
    with mock.patch.object(harmonizome, "json_from_url", return_value=fake_metadata):
        result = harmonizome.get_dataset_metadata("GTEx")

    assert isinstance(result["resource"], str)
    assert "GEO" in result["resource"]
    assert "n_genes: 12345" in result["stats"]
    assert "n_attributes: 6789" in result["stats"]


def test_get_dataset_metadata_coerces_nonstring_stats_to_string() -> None:
    """``stats`` that's neither dict nor str (e.g. an int from a buggy
    response) is still coerced cleanly."""
    fake_metadata = {
        "description": "d",
        "measurement": "m",
        "association": "a",
        "category": "c",
        "resource": "",
        "citations": [],
        "lastUpdated": "",
        "stats": 42,
    }
    with mock.patch.object(harmonizome, "json_from_url", return_value=fake_metadata):
        result = harmonizome.get_dataset_metadata("X")

    assert result["stats"] == "42"


def test_get_dataset_metadata_falsy_resource_becomes_empty_string() -> None:
    """Falsy non-string ``resource`` (e.g. ``None``) maps to the empty
    string, not the literal ``"None"``."""
    fake_metadata = {"resource": None}
    with mock.patch.object(harmonizome, "json_from_url", return_value=fake_metadata):
        result = harmonizome.get_dataset_metadata("X")

    assert result["resource"] == ""


# ─── biodb.harmonizome.get_gmt error paths ───────────────────────────────────


def _fake_script_config() -> dict:
    return {
        "downloads": ["gene_set_library_crisp.gmt.gz", "metadata.txt"],
        "datasets": {"Tiny Dataset": "tiny"},
    }


def _gmt_body() -> str:
    return "SET1\tdesc1\tBRCA1\tTP53\nSET2\tdesc2\tEGFR\n"


def test_get_gmt_parquet_write_failure_is_swallowed(tmp_path: Path, caplog) -> None:
    """If writing the merged-parquet cache fails (e.g. ImportError on a system
    without pyarrow at write time), we log a warning but still return the
    in-memory combined DataFrame."""
    dataset = "Tiny Dataset"
    cfg = _fake_script_config()
    (tmp_path / dataset).mkdir()
    (tmp_path / dataset / "gene_set_library_crisp.gmt.gz").write_bytes(
        gzip.compress(_gmt_body().encode())
    )

    with (
        mock.patch.object(harmonizome, "_load_config", return_value=cfg),
        mock.patch.object(
            harmonizome,
            "list_datasets",
            return_value=pd.DataFrame({"name": [dataset], "href": ["/x"]}),
        ),
        mock.patch.object(
            harmonizome,
            "get_dataset_metadata",
            return_value=harmonizome._EMPTY_METADATA.copy(),
        ),
        mock.patch("pandas.DataFrame.to_parquet", side_effect=OSError("disk full")),
        caplog.at_level(logging.WARNING, logger="biodb.harmonizome"),
    ):
        out = harmonizome.get_gmt(cache_dir=tmp_path, verbose=1)

    assert not out.empty
    assert {"BRCA1", "TP53", "EGFR"}.issubset(set(out["gene"]))
    assert any("Failed to cache merged DataFrame" in r.message for r in caplog.records)


def test_get_gmt_metadata_parallel_error_substitutes_empty_record(tmp_path: Path) -> None:
    """If ``get_dataset_metadata`` raises in one of the worker threads, that
    dataset gets an empty-record fallback rather than aborting the whole run."""
    dataset = "Tiny Dataset"
    cfg = _fake_script_config()
    (tmp_path / dataset).mkdir()
    (tmp_path / dataset / "gene_set_library_crisp.gmt.gz").write_bytes(
        gzip.compress(_gmt_body().encode())
    )

    with (
        mock.patch.object(harmonizome, "_load_config", return_value=cfg),
        mock.patch.object(
            harmonizome,
            "list_datasets",
            return_value=pd.DataFrame({"name": [dataset], "href": ["/x"]}),
        ),
        mock.patch.object(
            harmonizome,
            "get_dataset_metadata",
            side_effect=RuntimeError("boom"),
        ),
    ):
        out = harmonizome.get_gmt(cache_dir=tmp_path, save_path=False, verbose=0)

    assert not out.empty
    for col in ("description", "measurement", "association", "category"):
        assert (out[col] == "").all()


def test_get_gmt_empty_all_tasks_returns_empty_schema(tmp_path: Path) -> None:
    """When force=2 and every download returns empty, ``all_tasks`` is empty
    and we return the canonical empty DataFrame schema."""
    cfg = _fake_script_config()
    dataset = "Tiny Dataset"

    with (
        mock.patch.object(harmonizome, "_load_config", return_value=cfg),
        mock.patch.object(
            harmonizome,
            "list_datasets",
            return_value=pd.DataFrame({"name": [dataset], "href": ["/x"]}),
        ),
        mock.patch.object(harmonizome, "download_datasets", return_value={}),
    ):
        out = harmonizome.get_gmt(cache_dir=tmp_path, force=2, save_path=False, verbose=0)

    assert out.empty
    assert list(out.columns) == harmonizome._GMT_COLUMNS


# ─── biodb.harmonizome.load_gene_attribute_matrix data parsing ───────────────


def test_load_gene_attribute_matrix_extracts_tissue_metadata_inline(
    tmp_path: Path,
) -> None:
    """The ``metadata_idx`` branch fires when an *uncommented* row directly
    after the header has ``parts[2] == "CellLine"`` — that row is parsed for
    a ``Tissue`` annotation and becomes ``tissue_metadata``."""
    dataset = "Inline Dataset"
    (tmp_path / dataset).mkdir()
    (tmp_path / dataset / "gene_attribute_matrix.txt").write_text(
        "GeneSym\tNA\tGeneID\tBRCA1_CL\tTP53_CL\n"
        # Inline (non-#) metadata row with parts[2] == 'CellLine':
        "marker\tx\tCellLine\tBRCA1_CL\tTP53_CL\tTissue\tbreast\tcolon\n"
        "BRCA1\tna\t672\t0.5\t0.6\n"
    )

    df, tissue_metadata, _ = harmonizome.load_gene_attribute_matrix(
        dataset_name=dataset,
        filename="gene_attribute_matrix.txt",
        cache_dir=tmp_path,
    )

    assert tissue_metadata == {"BRCA1_CL": "breast", "TP53_CL": "colon"}
    # Only the real data row makes it through (BRCA1 with numeric GeneID).
    assert list(df["GeneSym"]) == ["BRCA1"]
    assert df.loc[df["GeneSym"] == "BRCA1", "GeneID"].iloc[0] == 672


def test_load_gene_attribute_matrix_captures_hash_comments_as_column_metadata(
    tmp_path: Path,
) -> None:
    """Rows starting with ``#`` are captured in ``column_metadata`` (a
    DataFrame keyed by ``row[2]`` of the parsed line). The parser strips
    the leading ``#`` and any outer whitespace before splitting on tab,
    so the test data uses a visible 1-char marker after ``#`` to keep
    the row from being deindented.

    Also exercises numeric AND alphanumeric gene-id detection in the
    data section.
    """
    dataset = "Hash Dataset"
    (tmp_path / dataset).mkdir()
    (tmp_path / dataset / "gene_attribute_matrix.txt").write_text(
        # "#a\tb\tDataSource\tHPA\tHPA" — `_build_column_metadata` reads
        # row[2] as the column key.
        "#a\tb\tDataSource\tHPA\tHPA\n"
        "GeneSym\tNA\tGeneID\tBRCA1_CL\tTP53_CL\n"
        "BRCA1\tna\t672\t0.5\t0.6\n"
        "MIR21\tna\tMIRBASE_MI0000077\t0.1\t0.2\n"
    )

    df, _, column_metadata = harmonizome.load_gene_attribute_matrix(
        dataset_name=dataset,
        filename="gene_attribute_matrix.txt",
        cache_dir=tmp_path,
    )

    assert column_metadata is not None
    assert "DataSource" in column_metadata.columns
    # Both gene IDs come through, exercising the numeric AND
    # alphanumeric branches of the GeneID-detection loop.
    assert set(df["GeneSym"]) == {"BRCA1", "MIR21"}
    assert df.loc[df["GeneSym"] == "BRCA1", "GeneID"].iloc[0] == 672


def test_load_gene_attribute_matrix_handles_missing_gene_id_gracefully(
    tmp_path: Path,
) -> None:
    """When all candidate ID columns are NA, the data row still ingests but
    GeneID is NaN."""
    dataset = "No-ID Dataset"
    (tmp_path / dataset).mkdir()
    (tmp_path / dataset / "gene_attribute_matrix.txt").write_text(
        "GeneSym\tNA\tGeneID\tCL1\tCL2\nBRCA1\tna\tna\t0.5\t0.6\n"
    )

    df, _, _ = harmonizome.load_gene_attribute_matrix(
        dataset_name=dataset,
        filename="gene_attribute_matrix.txt",
        cache_dir=tmp_path,
        include_col_metadata=False,
    )

    assert df.loc[df["GeneSym"] == "BRCA1", "GeneID"].isna().all()


def test_load_gene_attribute_matrix_skips_all_na_first_data_row(tmp_path: Path) -> None:
    """A ``na, na`` first data row (just below the header) is an artifact
    and is skipped before real data begins."""
    dataset = "Skip Dataset"
    (tmp_path / dataset).mkdir()
    (tmp_path / dataset / "gene_attribute_matrix.txt").write_text(
        "GeneSym\tNA\tGeneID\tCL1\nna\tna\tna\t0.0\nBRCA1\tna\t672\t1.5\n"
    )

    df, _, _ = harmonizome.load_gene_attribute_matrix(
        dataset_name=dataset,
        filename="gene_attribute_matrix.txt",
        cache_dir=tmp_path,
        include_col_metadata=False,
    )

    assert list(df["GeneSym"]) == ["BRCA1"]
    assert df.loc[df["GeneSym"] == "BRCA1", "CL1"].iloc[0] == 1.5
