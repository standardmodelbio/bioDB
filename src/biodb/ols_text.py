"""Complete OLS v2 textual annotations, including SKOS labels lost by legacy OLS.

No graph labels or logical axioms are converted into invented descriptions.
Resumable page caches permit large SNOMED downloads; the final cache must pass
count/ID audits before publication. Live data can change, so record provenance.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests

LOG = logging.getLogger(__name__)
V2_API = "https://www.ebi.ac.uk/ols4/api/v2"
SKOS = "http://www.w3.org/2004/02/skos/core#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
SYNONYM_PROPERTIES = [
    SKOS + "altLabel",
    "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym",
    "http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym",
    "http://www.geneontology.org/formats/oboInOwl#hasBroadSynonym",
    "http://www.geneontology.org/formats/oboInOwl#hasNarrowSynonym",
]
DEFINITION_PROPERTIES = [
    "http://purl.obolibrary.org/obo/IAO_0000115",
    SKOS + "definition",
    "http://www.geneontology.org/formats/oboInOwl#hasDefinition",
    "description",
    "definition",
]


def _strings(value):
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        return _strings(value.get("value"))
    if isinstance(value, list):
        return [text for item in value for text in _strings(item)]
    return []


def text_record(entity):
    """Normalize only this entity's literal annotations into legacy biodb columns."""
    labels = list(
        dict.fromkeys(_strings(entity.get("label")) + _strings(entity.get(RDFS + "label")))
    )
    preferred = _strings(entity.get(SKOS + "prefLabel"))
    label = next(iter(preferred or labels), "")
    synonyms = list(
        dict.fromkeys(
            [
                *labels,
                *preferred,
                *_strings(entity.get("synonyms")),
                *[s for key in SYNONYM_PROPERTIES for s in _strings(entity.get(key))],
            ]
        )
    )
    synonyms = [s for s in synonyms if s != label]
    definitions = list(
        dict.fromkeys(s for key in DEFINITION_PROPERTIES for s in _strings(entity.get(key)))
    )
    return {
        "obo_id": entity.get("curie") or entity.get("obo_id") or "",
        "label": label,
        "iri": entity.get("iri") or "",
        "description": definitions,
        "synonyms": synonyms,
        "is_obsolete": bool(entity.get("isObsolete", False)),
    }


def download_text_cache(
    ontology,
    destination,
    *,
    page_cache,
    size=500,
    workers=4,
    timeout=60,
    attempts=4,
    expected_ids=None,
):
    """Download and audit every v2 class, resuming compact per-page JSON caches.

    Cache directory must be specific to one ontology snapshot and page size.
    expected_ids optionally pins the concept universe to an existing snapshot;
    mismatches abort and leave the original snapshot untouched.
    """
    if size < 1 or not 1 <= workers <= 8:
        raise ValueError("size must be positive and workers must be between 1 and 8")
    destination, page_cache = Path(destination), Path(page_cache)
    page_cache.mkdir(parents=True, exist_ok=True)
    url = f"{V2_API}/ontologies/{ontology}/classes"
    metadata_url = f"{V2_API}/ontologies/{ontology}"
    metadata_response = requests.get(metadata_url, timeout=timeout)
    metadata_response.raise_for_status()
    ontology_metadata = metadata_response.json()
    source_metadata = {
        key: ontology_metadata.get(key)
        for key in [
            "ontologyId",
            "iri",
            "title",
            "loaded",
            "sourceFileTimestamp",
            "http://www.w3.org/2002/07/owl#versionIRI",
            "numberOfClasses",
        ]
    }
    started_at = datetime.now(timezone.utc).isoformat()
    settings_path = page_cache / "settings.json"
    settings = {"ontology": ontology, "url": url, "size": size, "source_metadata": source_metadata}
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("page cache settings differ from requested source")
    settings_path.write_text(json.dumps(settings))

    def fetch(page):
        path = page_cache / f"{page:06d}.json"
        if path.exists():
            return json.loads(path.read_text())
        for attempt in range(attempts):
            try:
                response = requests.get(url, params={"page": page, "size": size}, timeout=timeout)
                response.raise_for_status()
                payload = response.json()
                records = [text_record(row) for row in payload["elements"]]
                if not records or any(not row["obo_id"] for row in records):
                    raise ValueError(f"empty/unidentified OLS page {page}")
                data = {
                    "page": page,
                    "totalPages": payload["totalPages"],
                    "totalElements": payload["totalElements"],
                    "records": records,
                }
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(data))
                temporary.replace(path)
                return data
            except (requests.RequestException, ValueError, KeyError):
                if attempt + 1 == attempts:
                    raise
                time.sleep(min(2**attempt, 8))
        raise RuntimeError("unreachable")

    first = fetch(0)
    total_pages, expected_total = first["totalPages"], first["totalElements"]
    schema = pa.schema(
        [
            ("obo_id", pa.string()),
            ("label", pa.string()),
            ("iri", pa.string()),
            ("description", pa.list_(pa.string())),
            ("synonyms", pa.list_(pa.string())),
            ("is_obsolete", pa.bool_()),
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial.parquet")
    counts = dict(total=0, with_labels=0, with_synonyms=0, with_descriptions=0, obsolete=0)
    seen = set()
    with (
        pq.ParquetWriter(temporary, schema=schema) as writer,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        # Bounded windows prevent executor.map from buffering the entire ontology.
        for start in range(0, total_pages, workers * 2):
            for data in pool.map(fetch, range(start, min(start + workers * 2, total_pages))):
                if data["totalElements"] != expected_total or data["totalPages"] != total_pages:
                    raise ValueError("OLS concept universe changed during download")
                records = data["records"]
                for row in records:
                    if row["obo_id"] in seen:
                        raise ValueError("duplicate concept ID across pages")
                    seen.add(row["obo_id"])
                    counts["total"] += 1
                    counts["with_labels"] += bool(row["label"])
                    counts["with_synonyms"] += bool(row["synonyms"])
                    counts["with_descriptions"] += bool(row["description"])
                    counts["obsolete"] += row["is_obsolete"]
                writer.write_table(pa.Table.from_pylist(records, schema=schema))
            LOG.info(
                "OLS v2 pages %d/%d: %d concepts",
                min(start + workers * 2, total_pages),
                total_pages,
                counts["total"],
            )
    if len(seen) != expected_total:
        raise ValueError("incomplete OLS download")
    if expected_ids is not None and seen != set(expected_ids):
        raise ValueError("OLS IDs differ from pinned snapshot; choose a new release explicitly")
    final_response = requests.get(metadata_url, timeout=timeout)
    final_response.raise_for_status()
    final_metadata = final_response.json()
    if any(final_metadata.get(key) != value for key, value in source_metadata.items()):
        raise ValueError("OLS ontology metadata changed during download; discard mixed pages")
    temporary.replace(destination)
    provenance = {
        "api": "ols-v2",
        "url": url,
        "source_metadata": source_metadata,
        "download_started_at": started_at,
        "download_completed_at": datetime.now(timezone.utc).isoformat(),
        "coverage": counts,
        "text_properties": {"synonyms": SYNONYM_PROPERTIES, "definitions": DEFINITION_PROPERTIES},
        "all_labels_preserved": True,
        "logical_axioms_used_as_descriptions": False,
    }
    destination.with_suffix(".json").write_text(json.dumps(provenance, indent=2))
    return provenance
