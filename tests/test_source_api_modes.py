"""Live integration tests for the targeted-query "API mode" added to each
source module. These were 🚧 stubs in the Sources table until this PR.

Each test hits the real upstream API with a payload small enough to keep
the per-CI cost bounded (no full-corpus downloads). Mocked-only tests
would never catch upstream schema drift here, which is the recurring
failure mode for these specific services.
"""

from __future__ import annotations

import pandas as pd
import pytest

# ─── biodb.opentargets_graphql.query_drug — schema-drift canary ─────────────
# The other OT query helpers (query_target, query_disease) are exercised in
# test_opentargets.py; query_drug is the field most prone to schema rename
# (we lost ``isApproved`` and ``maximumClinicalTrialPhase`` in a 2026 schema
# bump). Hit the live endpoint so CI fails fast on the next rename.


def test_opentargets_query_drug_returns_known_chembl_record() -> None:
    """Aspirin (CHEMBL25) is the canonical "is the drug field set up
    properly" probe — small, public, never going away."""
    from biodb import opentargets_graphql as otg

    aspirin = otg.query_drug("CHEMBL25")
    assert aspirin["id"] == "CHEMBL25"
    assert aspirin["name"].lower() == "aspirin"
    # ``drugType`` is "Small molecule" or similar — just make sure it exists.
    assert aspirin.get("drugType")


# ─── biodb.monarch.query_* (BioLink v3 REST) ────────────────────────────────


def test_monarch_query_entity_returns_documented_fields() -> None:
    """BRCA1 (HGNC:1100) round-trips through the BioLink entity endpoint."""
    from biodb import monarch

    brca1 = monarch.query_entity("HGNC:1100")
    assert brca1["id"] == "HGNC:1100"
    assert brca1["name"] == "BRCA1"
    assert brca1["category"] == "biolink:Gene"
    # ``xref`` should include the Ensembl + OMIM crossrefs.
    assert any("ENSEMBL:ENSG" in x for x in brca1.get("xref", []))


def test_monarch_query_associations_returns_paginated_items() -> None:
    """Filter by subject; verify pagination metadata + association shape."""
    from biodb import monarch

    page = monarch.query_associations(subject="HGNC:1100", limit=3)
    assert page["limit"] == 3
    assert isinstance(page["items"], list)
    assert page["total"] > 100, "BRCA1 should have many associations"
    # Each item is a BioLink triple.
    item = page["items"][0]
    for col in ("subject", "predicate", "object", "subject_category"):
        assert col in item


def test_monarch_query_gene_associations_returns_dataframe() -> None:
    """The convenience wrapper aggregates pagination into a DataFrame."""
    from biodb import monarch

    df = monarch.query_gene_associations("HGNC:1100", limit=50)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 50, f"Expected pagination to fetch many rows, got {len(df)}"
    assert (df["subject"] == "HGNC:1100").all()


# ─── biodb.monarch.query_cypher (public Neo4j HTTP API) ─────────────────────


def test_monarch_query_cypher_count_nodes() -> None:
    """A trivial ``MATCH (n) RETURN count(n)`` round-trips through the
    public Neo4j HTTP transactional endpoint."""
    from biodb import monarch

    df = monarch.query_cypher("MATCH (n) RETURN count(n) AS total")
    assert list(df.columns) == ["total"]
    assert df.iloc[0]["total"] > 1_000_000, (
        f"Monarch KG should have well over 1 M nodes; got {df.iloc[0]['total']} — suspicious"
    )


def test_monarch_query_cypher_parameters_round_trip() -> None:
    """Parameter binding (``$id``) is the only safe way to pass a CURIE
    into a Cypher query — verify the wire format works end-to-end."""
    from biodb import monarch

    df = monarch.query_cypher(
        "MATCH (n {id: $id}) RETURN n.id AS id, n.name AS name LIMIT 1",
        parameters={"id": "HGNC:1100"},
    )
    assert len(df) == 1
    assert df.iloc[0]["id"] == "HGNC:1100"
    assert df.iloc[0]["name"] == "BRCA1"


def test_monarch_query_cypher_surfaces_syntax_errors() -> None:
    """Neo4j returns Cypher syntax errors inside a 200 response body's
    ``"errors"`` array. We have to surface those as Python exceptions."""
    from biodb import monarch

    with pytest.raises(RuntimeError, match="SyntaxError"):
        monarch.query_cypher("MATCH (n RETURN n")  # missing closing paren


def test_monarch_query_neighbors_returns_real_edges() -> None:
    """BRCA1 has many neighbours in the KG; the convenience wrapper
    should pull a sample with the documented columns."""
    from biodb import monarch

    df = monarch.query_neighbors("HGNC:1100", limit=5)
    assert len(df) == 5
    for col in ("predicate", "neighbor_id", "neighbor_name", "neighbor_category"):
        assert col in df.columns


# ─── biodb.clinvar.query_variant / query_gene (NCBI E-utilities) ────────────


def test_clinvar_query_variant_by_uid() -> None:
    """Resolve a numeric UID to its full ESummary record."""
    from biodb import clinvar

    record = clinvar.query_variant(12345)
    assert record["accession"] == "VCV000012345"
    # Title carries the HGVS-like coordinate + protein change.
    assert "NM_001065.4" in record["title"] or "VCV" in record["title"]


def test_clinvar_query_variant_by_accession() -> None:
    """Pass a VCV accession; the helper resolves it to a UID first."""
    from biodb import clinvar

    record = clinvar.query_variant("VCV000012345")
    assert record["accession"] == "VCV000012345"


def test_clinvar_query_gene_returns_uids() -> None:
    """``BRCA1[gene]`` resolves to thousands of UIDs; we just check shape."""
    from biodb import clinvar

    uids = clinvar.query_gene("BRCA1", retmax=5)
    assert isinstance(uids, list)
    assert len(uids) == 5
    assert all(u.isdigit() for u in uids)


def test_clinvar_query_variant_raises_on_unknown_accession() -> None:
    """An unmappable accession surfaces a clean ``KeyError``."""
    from biodb import clinvar

    with pytest.raises(KeyError, match="not found"):
        clinvar.query_variant("VCV99999999999")


# ─── biodb.msigdb.query_gene_set / query_genes ──────────────────────────────


def test_msigdb_query_gene_set_returns_record() -> None:
    """Hallmark Apoptosis set round-trips with documented fields."""
    from biodb import msigdb

    record = msigdb.query_gene_set("HALLMARK_APOPTOSIS")
    assert record["systematicName"] == "M5902"
    assert isinstance(record["geneSymbols"], list)
    # Hallmark Apoptosis has ~160 genes.
    assert 100 < len(record["geneSymbols"]) < 300


def test_msigdb_query_genes_returns_just_symbols() -> None:
    """``query_genes`` flattens the response to the gene-symbol list."""
    from biodb import msigdb

    genes = msigdb.query_genes("HALLMARK_HYPOXIA")
    assert isinstance(genes, list)
    assert all(isinstance(g, str) for g in genes)
    # HIF1A is a canonical hypoxia gene; verify the set is what we think.
    assert "HIF1A" in genes or "VEGFA" in genes


def test_msigdb_query_gene_set_raises_on_unknown() -> None:
    """Unknown set → ``KeyError`` (MSigDB serves an HTML 200 for unknowns,
    not a 404 — we sniff the response shape rather than the status code)."""
    from biodb import msigdb

    with pytest.raises(KeyError, match="not found"):
        msigdb.query_gene_set("HALLMARK_NONEXISTENT_FAKE_SET")


# ─── biodb.gwas_atlas.query_trait / list_traits ─────────────────────────────


def test_gwas_atlas_query_trait_by_substring(tmp_path) -> None:
    """``query_trait("Alzheimer")`` returns multiple AD-related studies."""
    from biodb import gwas_atlas

    # Reset module-level cache so cache_dir is honoured.
    gwas_atlas._METADATA_CACHE = None
    rows = gwas_atlas.query_trait("Alzheimer", cache_dir=tmp_path)
    assert isinstance(rows, pd.DataFrame)
    assert len(rows) > 0
    # Every returned row should mention Alzheimer in the Trait field.
    assert rows["Trait"].astype(str).str.contains("Alzheimer", case=False).all()


def test_gwas_atlas_query_trait_by_numeric_id(tmp_path) -> None:
    """A numeric lookup hits ``id`` first; the smallest study (id=1) exists."""
    from biodb import gwas_atlas

    gwas_atlas._METADATA_CACHE = None
    rows = gwas_atlas.query_trait(1, cache_dir=tmp_path)
    assert isinstance(rows, pd.DataFrame)
    assert len(rows) >= 1


def test_gwas_atlas_query_trait_explicit_column(tmp_path) -> None:
    """``column="PMID"`` forces the lookup column. (Note: PMID is stored as
    a string in the metadata TSV, so the comparison stays str-shaped.)"""
    from biodb import gwas_atlas

    gwas_atlas._METADATA_CACHE = None
    # First peek at the metadata to grab a real PMID.
    meta = gwas_atlas.list_traits(cache_dir=tmp_path)
    sample_pmid = str(meta["PMID"].dropna().iloc[0])

    rows = gwas_atlas.query_trait(sample_pmid, column="PMID", cache_dir=tmp_path)
    assert len(rows) >= 1
    assert (rows["PMID"].astype(str) == sample_pmid).all()


def test_gwas_atlas_query_trait_rejects_unknown_column(tmp_path) -> None:
    """A bad ``column=`` argument fails fast with a useful error."""
    from biodb import gwas_atlas

    gwas_atlas._METADATA_CACHE = None
    with pytest.raises(KeyError, match="Column"):
        gwas_atlas.query_trait("anything", column="not_a_column", cache_dir=tmp_path)


# ─── biodb.uniprot bulk FASTA — UI smoke only, no full download ─────────────


def test_uniprot_swissprot_fasta_url_resolves_offline() -> None:
    """Verify the constant points at a real FTP path. We don't download
    Swiss-Prot here (~90 MB); the URL HEAD is enough to catch a path
    change. Use ``download_swissprot_fasta`` in pipelines that need
    the file."""
    import requests

    from biodb import uniprot

    url = f"{uniprot.UNIPROT_FTP_BASE_URL}/{uniprot.SWISSPROT_FASTA_FILENAME}"
    response = requests.head(url, timeout=10, allow_redirects=True)
    assert response.status_code == 200, (
        f"UniProt Swiss-Prot FASTA URL {url} returned {response.status_code}"
    )
    # Should report a sane content-length (Swiss-Prot is roughly 90 MB).
    if "content-length" in response.headers:
        size = int(response.headers["content-length"])
        assert size > 50_000_000, f"Swiss-Prot reported size {size} bytes — suspiciously small"


def test_uniprot_iter_fasta_records_reads_local_file(tmp_path) -> None:
    """``iter_fasta_records`` can stream from a local FASTA file. Tests the
    iteration path against synthetic data so we don't have to download
    Swiss-Prot just to verify the iterator works."""
    pytest.importorskip("Bio")
    from biodb import uniprot

    # Write two FASTA records.
    fasta = tmp_path / "tiny.fasta"
    fasta.write_text(">sp|P12345|TEST_FAKE\nMSEQUENCEA\n>sp|P67890|TEST_FAKE2\nMSEQUENCEB\n")

    records = list(uniprot.iter_fasta_records(fasta_path=fasta))
    assert len(records) == 2
    assert records[0].id == "sp|P12345|TEST_FAKE"
    assert str(records[0].seq) == "MSEQUENCEA"


def test_uniprot_count_records_streaming(tmp_path) -> None:
    """``count_swissprot_records`` is the streaming-count probe."""
    pytest.importorskip("Bio")
    from biodb import uniprot

    fasta = tmp_path / "tiny.fasta"
    fasta.write_text(">a\nM\n>b\nM\n>c\nM\n")
    assert uniprot.count_swissprot_records(fasta_path=fasta) == 3
