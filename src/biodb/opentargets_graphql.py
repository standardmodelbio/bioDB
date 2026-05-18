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
