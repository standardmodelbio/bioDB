"""Targeted OpenTargets Platform GraphQL queries (API mode).

Complements :mod:`biodb.opentargets`'s bulk-download (FTP-mode) helpers with
one-record-at-a-time GraphQL lookups. Use this when you need a fresh
single-target / single-disease / single-drug / single-variant payload and
don't want to download an entire Parquet release.

The ``query_*`` wrappers issue deep, multi-level GraphQL queries that pull
back the rich nested payload OT exposes (associated targets, drug
candidates, HPO phenotypes, MoA / indications, ClinVar + UniProt
evidence, credible sets, pharmacogenomics, literature occurrences, ...)
-- not just the shallow top-level scalars. Each wrapper exposes
size knobs (``assoc_size``, ``ae_size``, ``ev_size``, ``cs_size``,
``lit_size``) for the per-section pagination caps.

Endpoint
--------
- GraphQL endpoint: https://api.platform.opentargets.org/api/v4/graphql
- Raw SDL schema (text): https://api.platform.opentargets.org/api/v4/graphql/schema
- GraphQL Playground: https://api.platform.opentargets.org/api/v4/graphql/playground

OpenTargets docs and AI tooling
-------------------------------
- GraphQL API user guide: https://platform-docs.opentargets.org/data-access/graphql-api
- Official OT Platform MCP server (https://mcp.platform.opentargets.org/mcp)
  -- exposes schema-introspection + query tools so LLMs can craft well-formed
  queries directly. See https://platform-docs.opentargets.org/data-access/model-context-protocol
  and source at https://github.com/opentargets/open-targets-platform-mcp.

Identifier formats
------------------
- **Targets / Genes**: Ensembl IDs (e.g. ``"ENSG00000012048"`` for BRCA1).
- **Diseases**: EFO / MONDO / HP IDs in underscored form
  (``"EFO_0004911"``, ``"MONDO_0007254"``).
  Colon-form (``"MONDO:0007254"``) is auto-normalised.
- **Drugs**: ChEMBL drug IDs (``"CHEMBL25"``). Parent / child molecules
  available via ``parentMolecule`` / ``childMolecules``.
- **Variants**: OT variant IDs in ``"chr_pos_ref_alt"`` format
  (``"19_11100252_C_T"``). For rsID → OT-id resolution use a top-level
  ``search`` query.

Examples
--------
>>> from biodb.opentargets_graphql import query_target
>>> brca1 = query_target("ENSG00000012048")  # doctest: +SKIP
>>> brca1["approvedSymbol"]  # doctest: +SKIP
'BRCA1'
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
import polars as pl

logger = logging.getLogger(__name__)

OT_GRAPHQL_API = "https://api.platform.opentargets.org/api/v4/graphql"
"""Open Targets Platform v4 GraphQL endpoint."""

_DEFAULT_TIMEOUT_S: float = 30.0
_DEFAULT_MAX_RETRIES: int = 3
_DEFAULT_BACKOFF_S: float = 1.0


def graphql_post(
    query: str,
    variables: dict[str, Any],
    *,
    endpoint: str = OT_GRAPHQL_API,
    client: httpx.Client | None = None,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    backoff_s: float = _DEFAULT_BACKOFF_S,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """POST a GraphQL query with exponential-backoff retries; return the ``data`` block.

    Parameters
    ----------
    query : str
        GraphQL query string.
    variables : dict
        GraphQL variables.
    endpoint : str
        GraphQL endpoint URL. Defaults to OT Platform v4.
    client : httpx.Client, optional
        Reusable HTTP client for connection pooling. If ``None``, one is
        created per call.
    max_retries : int, default 3
        Maximum retry attempts on transient failures.
    backoff_s : float, default 1.0
        Initial backoff in seconds; doubles each retry.
    timeout_s : float, default 30.0
        Per-request timeout.

    Returns
    -------
    dict
        The ``data`` field of the GraphQL response.

    Raises
    ------
    httpx.HTTPError
        If all retries fail.
    RuntimeError
        If the GraphQL response carries an ``errors`` block.

    Examples
    --------
    >>> graphql_post(  # doctest: +SKIP
    ...     "query Q($id: String!) { target(ensemblId: $id) { approvedSymbol } }",
    ...     {"id": "ENSG00000012048"},
    ... )
    {'target': {'approvedSymbol': 'BRCA1'}}
    """
    owned_client = client is None
    client = client or httpx.Client(timeout=timeout_s)
    last_err: Exception | None = None
    try:
        for attempt in range(max_retries):
            try:
                resp = client.post(
                    endpoint, json={"query": query, "variables": variables}, timeout=timeout_s
                )
                resp.raise_for_status()
                body = resp.json()
                if errors := body.get("errors"):
                    raise RuntimeError(f"OpenTargets GraphQL error: {errors}")
                return body["data"]
            except (httpx.HTTPError, RuntimeError) as exc:
                last_err = exc
                if attempt < max_retries - 1:
                    time.sleep(backoff_s * (2**attempt))
        assert last_err is not None
        raise last_err
    finally:
        if owned_client:
            client.close()


_TARGET_QUERY = """
query Target($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    approvedName
    biotype
    functionDescriptions
    genomicLocation { chromosome start end strand }
    proteinIds { id source }
    symbolSynonyms { label source }
    nameSynonyms { label source }
    subcellularLocations { location source }
    pathways { pathway pathwayId topLevelTerm }
  }
}
"""
# Note: the ``go { term aspect source }`` field was removed from
# Open Targets' Target GraphQL type sometime before 2026-05; pulling
# Gene Ontology annotations now requires a separate API. Dropping the
# field here keeps the query well-formed against the current schema.

_DISEASE_QUERY = """
query Disease($efoId: String!, $assocSize: Int!, $phenoSize: Int!) {
  disease(efoId: $efoId) {
    id
    name
    description
    isTherapeuticArea
    dbXRefs
    synonyms { terms relation }
    therapeuticAreas { id name }
    ancestors
    descendants
    parents { id name }
    children { id name }
    phenotypes(page: {index: 0, size: $phenoSize}) {
      count
      rows {
        phenotypeHPO { id name description namespace }
        evidence {
          aspect
          frequency
          evidenceType
          resource
          bioCuration
          diseaseFromSource
        }
      }
    }
    associatedTargets(page: {index: 0, size: $assocSize}) {
      count
      rows {
        score
        noveltyDirect
        noveltyIndirect
        datatypeScores { id score }
        target { id approvedSymbol approvedName biotype }
      }
    }
    drugAndClinicalCandidates {
      count
      rows {
        id
        maxClinicalStage
        drug { id name drugType maximumClinicalStage }
      }
    }
    literatureOcurrences {
      count
      rows { pmid pmcid publicationDate }
    }
  }
}
"""

_DRUG_QUERY = """
query Drug($chemblId: String!, $aeSize: Int!) {
  drug(chemblId: $chemblId) {
    id
    name
    description
    drugType
    maximumClinicalStage
    tradeNames
    synonyms
    crossReferences { source ids }
    drugWarnings {
      id
      warningType
      description
      toxicityClass
      country
      year
      efoTerm
      efoId
      chemblIds
      references { source url }
    }
    mechanismsOfAction {
      uniqueActionTypes
      uniqueTargetTypes
      rows {
        mechanismOfAction
        actionType
        targetName
        targets { id approvedSymbol approvedName }
        references { source ids urls }
      }
    }
    indications {
      count
      rows {
        id
        maxClinicalStage
        disease { id name therapeuticAreas { id name } }
        clinicalReports {
          id
          trialPhase
          trialOverallStatus
          trialStudyType
          source
          url
          year
          title
        }
      }
    }
    adverseEvents(page: {index: 0, size: $aeSize}) {
      count
      criticalValue
      rows { name meddraCode count logLR }
    }
    literatureOcurrences {
      count
      rows { pmid pmcid publicationDate }
    }
  }
}
"""

_VARIANT_QUERY = """
query Variant($variantId: String!, $evSize: Int!, $csSize: Int!) {
  variant(variantId: $variantId) {
    id
    chromosome
    position
    referenceAllele
    alternateAllele
    rsIds
    hgvsId
    variantDescription
    mostSevereConsequence { id label }
    alleleFrequencies { populationName alleleFrequency }
    variantEffect {
      method
      assessment
      assessmentFlag
      score
      normalisedScore
      target { id approvedSymbol }
    }
    transcriptConsequences {
      transcriptId
      isEnsemblCanonical
      impact
      aminoAcidChange
      consequenceScore
      codons
      lofteePrediction
      siftPrediction
      polyphenPrediction
      uniprotAccessions
      target { id approvedSymbol biotype }
      variantConsequences { id label }
    }
    proteinCodingCoordinates(page: {index: 0, size: 25}) {
      count
      rows {
        referenceAminoAcid
        alternateAminoAcid
        aminoAcidPosition
        uniprotAccessions
        target { id approvedSymbol }
      }
    }
    pharmacogenomics {
      drugs { drugFromSource drugId }
      phenotypeText
      genotypeAnnotationText
      pgxCategory
      evidenceLevel
      genotype
      variantRsId
      variantFunctionalConsequence { id label }
    }
    credibleSets(page: {index: 0, size: $csSize}) {
      count
      rows {
        studyId
        studyType
        beta
        standardError
        pValueMantissa
        pValueExponent
        qtlGeneId
        confidence
        finemappingMethod
        study { traitFromSource projectId studyType }
      }
    }
    evidences(
      datasourceIds: ["eva", "eva_somatic", "uniprot_variants", "uniprot_literature"]
      size: $evSize
    ) {
      count
      rows {
        datasourceId
        datatypeId
        clinicalSignificances
        confidence
        diseaseFromSource
        diseaseFromSourceId
        studyId
        allelicRequirements
        literature
        urls { niceName url }
        targetFromSourceId
        disease { id name }
        target { id approvedSymbol }
        variantFunctionalConsequence { id label }
      }
    }
  }
}
"""


def query_target(
    ensembl_id: str,
    *,
    client: httpx.Client | None = None,
) -> dict[str, Any] | None:
    """Fetch a single target by Ensembl gene ID.

    Parameters
    ----------
    ensembl_id : str
        Ensembl stable gene ID (e.g. ``"ENSG00000012048"``).
    client : httpx.Client, optional
        Reusable HTTP client.

    Returns
    -------
    dict or None
        The ``target`` object from the GraphQL response, or ``None`` if not found.

    Examples
    --------
    >>> brca1 = query_target("ENSG00000012048")  # doctest: +SKIP
    >>> brca1["approvedSymbol"]  # doctest: +SKIP
    'BRCA1'
    """
    data = graphql_post(_TARGET_QUERY, {"ensemblId": ensembl_id}, client=client)
    return data.get("target")


def query_disease(
    efo_id: str,
    *,
    assoc_size: int = 25,
    pheno_size: int = 25,
    client: httpx.Client | None = None,
) -> dict[str, Any] | None:
    """Fetch a single disease by EFO/MONDO ID with full nested context.

    The payload now includes ontology relatives (parents/children/ancestors/
    descendants), HPO phenotypes (with evidence rows), genetically-associated
    targets (with datatype-broken-down scores), and clinical drug candidates
    -- not just the name/description/synonyms a shallow query would return.

    Parameters
    ----------
    efo_id : str
        Disease ID in OT's internal underscored form (``"EFO_0000305"``,
        ``"MONDO_0007254"``, etc.). Colons are accepted and normalized.
    assoc_size : int, default 25
        Page size for ``associatedTargets``. OT caps at 100.
    pheno_size : int, default 25
        Page size for ``phenotypes``. OT caps at 100.
    client : httpx.Client, optional

    Returns
    -------
    dict or None

    Examples
    --------
    >>> bc = query_disease("MONDO_0007254")  # doctest: +SKIP
    >>> bc["name"]  # doctest: +SKIP
    'breast carcinoma'
    >>> len(bc["associatedTargets"]["rows"])  # doctest: +SKIP
    25
    """
    efo_id = efo_id.replace(":", "_")
    data = graphql_post(
        _DISEASE_QUERY,
        {"efoId": efo_id, "assocSize": assoc_size, "phenoSize": pheno_size},
        client=client,
    )
    return data.get("disease")


def query_drug(
    chembl_id: str,
    *,
    ae_size: int = 25,
    client: httpx.Client | None = None,
) -> dict[str, Any] | None:
    """Fetch a single drug by ChEMBL ID with full nested context.

    The payload now includes mechanisms of action (with target rows),
    indications (with linked diseases + max clinical stage), adverse events
    (with logLR + counts), drug warnings, and cross-references -- not just
    the shallow ``name``/``description``/``drugType`` block a minimal query
    would return.

    Parameters
    ----------
    chembl_id : str
        ChEMBL drug ID (``"CHEMBL25"``).
    ae_size : int, default 25
        Page size for ``adverseEvents``. OT caps at 100.
    client : httpx.Client, optional

    Returns
    -------
    dict or None

    Examples
    --------
    >>> aspirin = query_drug("CHEMBL25")  # doctest: +SKIP
    >>> aspirin["name"]  # doctest: +SKIP
    'ASPIRIN'
    >>> aspirin["mechanismsOfAction"]["rows"][0]["mechanismOfAction"]  # doctest: +SKIP
    'Cyclooxygenase inhibitor'
    """
    data = graphql_post(
        _DRUG_QUERY,
        {"chemblId": chembl_id, "aeSize": ae_size},
        client=client,
    )
    return data.get("drug")


def query_variant(
    variant_id: str,
    *,
    ev_size: int = 50,
    cs_size: int = 25,
    client: httpx.Client | None = None,
) -> dict[str, Any] | None:
    """Fetch a single variant by OT variant ID with full nested context.

    The payload now includes ClinVar + UniProt evidence rows (via
    ``evidences(datasourceIds: ["eva", "uniprot_variants", "uniprot_literature"])``),
    pharmacogenomics rows, and GWAS / molQTL credible sets containing
    this variant -- not just the bare allele / consequence block.

    Parameters
    ----------
    variant_id : str
        OT variant ID in ``"chr_pos_ref_alt"`` format (``"19_11100252_C_T"``).
        Use :func:`biodb.opentargets_graphql.graphql_post` with the OT
        ``search`` query to resolve from rsId.
    ev_size : int, default 50
        Page size for ``evidences``. OT caps at 100.
    cs_size : int, default 25
        Page size for ``credibleSets``. OT caps at 100.
    client : httpx.Client, optional

    Returns
    -------
    dict or None

    Examples
    --------
    >>> v = query_variant("19_11100252_C_T")  # doctest: +SKIP
    >>> v["rsIds"]  # doctest: +SKIP
    ['rs121908024']
    """
    data = graphql_post(
        _VARIANT_QUERY,
        {"variantId": variant_id, "evSize": ev_size, "csSize": cs_size},
        client=client,
    )
    return data.get("variant")


_MAP_IDS_QUERY = """
query MapIds($terms: [String!]!, $entities: [String!]!) {
  mapIds(queryTerms: $terms, entityNames: $entities) {
    mappings { term hits { id name entity } }
  }
}
"""
"""Free-text / symbol -> Ensembl ID lookup. ``mapIds`` is OT's public
fuzzy-search entrypoint; restricting via ``entityNames`` keeps the
result set to the kind of record the caller actually wants
(``"target"`` for gene lookups, ``"disease"`` for EFO/MONDO, etc.)."""


def map_symbols_to_ensembl(
    gene_symbols: list[str],
    *,
    client: httpx.Client | None = None,
) -> dict[str, str]:
    """Map HGNC gene symbols to Ensembl gene IDs via OT's ``mapIds`` query.

    Useful for panel construction where you have a list of HGNC
    symbols and need their canonical Ensembl IDs before any of the
    other Open Targets helpers (which all key on Ensembl).

    The wire format returns a list of ``{term, hits: [{id, name, entity}]}``
    entries; this helper takes the first hit per term and only keeps
    targets that actually resolved. Symbols with no hit are omitted
    rather than returning ``None`` -- callers should check ``in``
    membership of the returned dict.

    Parameters
    ----------
    gene_symbols : list[str]
        HGNC symbols (``["BRCA1", "TP53", ...]``). Case-insensitive
        on OT's side but the input case is preserved as the dict key.
    client : httpx.Client, optional
        Reusable client for batched workflows.

    Returns
    -------
    dict[str, str]
        ``{symbol: ensembl_id}`` for symbols with at least one hit.

    Examples
    --------
    >>> map_symbols_to_ensembl(["BRCA1"])["BRCA1"]  # doctest: +SKIP
    'ENSG00000012048'
    """
    data = graphql_post(
        _MAP_IDS_QUERY,
        {"terms": list(gene_symbols), "entities": ["target"]},
        client=client,
    )
    return {
        m["term"]: m["hits"][0]["id"]
        for m in data.get("mapIds", {}).get("mappings", [])
        if m.get("hits")
    }


_TARGET_ASSOC_DISEASES_QUERY = """
query TargetAssociatedDiseases($ensemblId: String!, $size: Int!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    associatedDiseases(page: {index: 0, size: $size}) {
      count
      rows {
        score
        datatypeScores { id score }
        disease { id name therapeuticAreas { id name } }
      }
    }
  }
}
"""
"""Focused target -> associated-diseases query. ``query_target`` already
exists but returns the metadata-only payload; this one is for the
"give me one gene's disease association scores" use case (panel
scoring, prioritisation pipelines). Splitting into a separate helper
keeps the ``query_target`` payload sane and skips the rest of the
target metadata when the caller only wants disease scores."""


def target_associated_diseases(
    ensembl_id: str,
    *,
    size: int = 200,
    client: httpx.Client | None = None,
) -> dict[str, Any] | None:
    """Fetch a target's ``associatedDiseases`` rows by Ensembl gene ID.

    Returns the ``target`` envelope (``id``, ``approvedSymbol``,
    ``associatedDiseases.{count, rows}``) so the caller can filter
    rows by disease ID or aggregate scores client-side. The full
    ``query_target`` payload is more than what most callers need for
    panel scoring; this query just pulls the disease rows.

    Parameters
    ----------
    ensembl_id : str
        Ensembl stable gene ID (e.g. ``"ENSG00000012048"``).
    size : int, default 200
        Page size for ``associatedDiseases``. OT caps individual
        pages at 200 -- bump only if you've split a target across
        the pagination boundary (rare; most targets have <100
        associated diseases above the OT relevance threshold).
    client : httpx.Client, optional
        Reusable client for batched workflows.

    Returns
    -------
    dict or None
        ``{"id": ..., "approvedSymbol": ..., "associatedDiseases": {"count": N, "rows": [...]}}``
        or ``None`` if no such target exists.

    Examples
    --------
    >>> brca1 = target_associated_diseases("ENSG00000012048")  # doctest: +SKIP
    >>> brca1["approvedSymbol"]  # doctest: +SKIP
    'BRCA1'
    >>> brca1["associatedDiseases"]["count"] > 0  # doctest: +SKIP
    True
    """
    data = graphql_post(
        _TARGET_ASSOC_DISEASES_QUERY,
        {"ensemblId": ensembl_id, "size": size},
        client=client,
    )
    return data.get("target")


# ---------------------------------------------------------------------------
# Panel scoring
#
# Composes the symbol -> Ensembl resolution + per-target associated-disease
# queries above with EFO/MONDO/HP id normalisation and polars-DataFrame
# aggregation. Mirrors the API the seqlab panel builder previously hosted
# under ``seqlab.panel.opentargets``; consolidated here in biodb because the
# building blocks live in this module and there are no seqlab-specific
# concerns left.
# ---------------------------------------------------------------------------


def _normalise_id(disease_id: str) -> str:
    """Normalise an ontology ID to OpenTargets' internal underscored form.

    OpenTargets uses ``MONDO_0007254`` not ``MONDO:0007254`` and
    ``EFO_0000305`` not ``EFO:0000305``. Inputs already in underscored
    form are returned unchanged. Module-private (not re-exported).

    Parameters
    ----------
    disease_id : str
        EFO / MONDO / HP id in either ``"EFO:0000305"`` or
        ``"EFO_0000305"`` form.

    Returns
    -------
    str
        The same id with any ``:`` separator replaced by ``_``.

    Raises
    ------
    AttributeError
        If ``disease_id`` is not a string.

    Examples
    --------
    >>> _normalise_id("EFO:0000305")
    'EFO_0000305'
    >>> _normalise_id("EFO_0000305")
    'EFO_0000305'
    >>> _normalise_id("MONDO:0007254")
    'MONDO_0007254'
    """
    return disease_id.replace(":", "_")


def fetch_panel_scores(
    gene_symbols: list[str],
    disease_ids: list[str],
    *,
    client: httpx.Client | None = None,
) -> pl.DataFrame:
    """Fetch OpenTargets association scores per gene x disease pair.

    Resolves each HGNC symbol to its Ensembl gene ID via
    :func:`map_symbols_to_ensembl`, pulls the ``associatedDiseases`` rows
    via :func:`target_associated_diseases`, and keeps only rows whose
    ``disease.id`` is in the normalised ``disease_ids`` set. Per-gene
    HTTP failures are logged at WARNING and the gene is skipped (the
    rest of the panel still scores). Symbols that fail to resolve to
    an Ensembl id are silently dropped.

    Parameters
    ----------
    gene_symbols : list[str]
        HGNC symbols (``["BRCA1", "TP53", ...]``). Case is preserved in
        any returned ``gene_symbol`` values but OT's resolver is
        case-insensitive.
    disease_ids : list[str]
        EFO / MONDO / HP ids in either ``"EFO_0000305"`` or
        ``"EFO:0000305"`` form -- colon form is normalised internally.
    client : httpx.Client, optional
        Reusable HTTP client for connection pooling across the resolve +
        per-gene fetch passes. If ``None``, one is created for the call.

    Returns
    -------
    polars.DataFrame
        Columns: ``gene_id`` (Utf8), ``gene_symbol`` (Utf8),
        ``cancer_type`` (Utf8 -- the matched normalised disease id),
        ``ot_score`` (Float64), ``ot_evidence`` (Utf8 -- the matched
        disease display name). One row per ``(gene, matched_disease)``
        pair. The empty-result DataFrame still carries the full schema
        so downstream group-by operations don't need to guard on it.

    Raises
    ------
    httpx.HTTPError
        Only from the initial ``map_symbols_to_ensembl`` call -- per-gene
        association failures are swallowed and logged.

    Examples
    --------
    >>> df = fetch_panel_scores(["BRCA1"], ["EFO_0000305"])  # doctest: +SKIP
    >>> df.height >= 1  # doctest: +SKIP
    True
    """
    wanted = {_normalise_id(d) for d in disease_ids}
    rows: list[dict[str, Any]] = []
    owns_client = client is None
    if client is None:
        client = httpx.Client()
    try:
        mapping = map_symbols_to_ensembl(gene_symbols, client=client)
        for sym in gene_symbols:
            ensembl_id = mapping.get(sym)
            if ensembl_id is None:
                continue
            try:
                target = target_associated_diseases(ensembl_id, client=client)
            except (httpx.HTTPError, RuntimeError) as exc:
                logger.warning(
                    "%s (%s) associated-diseases fetch failed: %s; skipping",
                    sym,
                    ensembl_id,
                    exc,
                )
                continue
            if not target:
                continue
            approved = target.get("approvedSymbol", sym)
            for r in target["associatedDiseases"]["rows"]:
                did = _normalise_id(r["disease"]["id"])
                if did not in wanted:
                    continue
                rows.append(
                    {
                        "gene_id": ensembl_id,
                        "gene_symbol": approved,
                        "cancer_type": did,
                        "ot_score": float(r["score"]),
                        "ot_evidence": r["disease"]["name"],
                    }
                )
    finally:
        if owns_client:
            client.close()
    if not rows:
        return pl.DataFrame(
            schema={
                "gene_id": pl.Utf8,
                "gene_symbol": pl.Utf8,
                "cancer_type": pl.Utf8,
                "ot_score": pl.Float64,
                "ot_evidence": pl.Utf8,
            }
        )
    return pl.DataFrame(rows)


def fetch_aggregated_panel_scores(
    gene_symbols: list[str],
    disease_ids: list[str],
    *,
    client: httpx.Client | None = None,
) -> pl.DataFrame:
    """Fetch and aggregate OpenTargets scores to one row per gene.

    Thin wrapper around :func:`fetch_panel_scores` that group-bys on
    ``(gene_id, gene_symbol)`` and collapses the per-disease rows into:

    - ``ot_score`` -- max score across matched diseases
    - ``cancer_types`` -- list of matched (normalised) disease ids
    - ``diseases`` -- list of matched disease display names

    Order within the list columns mirrors the iteration order of
    :func:`fetch_panel_scores` (the OT row order for each gene); no
    sort is applied.

    Parameters
    ----------
    gene_symbols : list[str]
        HGNC symbols, forwarded to :func:`fetch_panel_scores`.
    disease_ids : list[str]
        EFO / MONDO / HP ids -- colon form auto-normalised.
    client : httpx.Client, optional
        Reusable HTTP client forwarded to :func:`fetch_panel_scores`.

    Returns
    -------
    polars.DataFrame
        Columns: ``gene_id`` (Utf8), ``gene_symbol`` (Utf8),
        ``ot_score`` (Float64), ``cancer_types`` (List[Utf8]),
        ``diseases`` (List[Utf8]). Zero rows if no gene matched any
        requested disease.

    Examples
    --------
    >>> df = fetch_aggregated_panel_scores(  # doctest: +SKIP
    ...     ["BRCA1"], ["EFO_0000305", "EFO_0001075"]
    ... )
    >>> df.columns  # doctest: +SKIP
    ['gene_id', 'gene_symbol', 'ot_score', 'cancer_types', 'diseases']
    """
    per_disease = fetch_panel_scores(gene_symbols, disease_ids, client=client)
    if per_disease.height == 0:
        return pl.DataFrame(
            schema={
                "gene_id": pl.Utf8,
                "gene_symbol": pl.Utf8,
                "ot_score": pl.Float64,
                "cancer_types": pl.List(pl.Utf8),
                "diseases": pl.List(pl.Utf8),
            }
        )
    return per_disease.group_by(["gene_id", "gene_symbol"]).agg(
        pl.col("ot_score").max(),
        pl.col("cancer_type").alias("cancer_types"),
        pl.col("ot_evidence").alias("diseases"),
    )
