"""Acquire and parse upstream OT constants; rendering belongs to consumers."""

from __future__ import annotations

import base64
import contextlib
import re
from pathlib import Path
from typing import Any

import requests

from biodb import _response_cache as cache

OT_REPO = "opentargets/ot-ui-apps"
OT_VARIANT_TS = "packages/ot-constants/src/variant.ts"
OT_PROFILE_HEADER_TSX = "apps/platform/src/pages/VariantPage/ProfileHeader.tsx"
OT_INDEX_TS = "packages/ot-constants/src/index.ts"


def fetch_source(
    path: str,
    *,
    ref: str = "main",
    cache_dir: str | Path | None = None,
    session: Any = None,
    timeout: float = 30.0,
    force: bool = False,
) -> tuple[str, str]:
    """Fetch GitHub source text and blob SHA, retaining validated raw responses."""
    url = f"https://api.github.com/repos/{OT_REPO}/contents/{path}"
    identity = url + "?ref=" + ref
    fp = cache.cache_path(cache_dir, "opentargets/constants", identity)
    with cache.locked(fp):
        data = None
        if not force:
            with contextlib.suppress(KeyError):
                data = cache.read(fp, identity)
        if data is None:
            response = (session or requests).get(url, params={"ref": ref}, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            text = base64.b64decode(data["content"], validate=False).decode()
            if not data.get("sha") or not text:
                raise ValueError("Malformed GitHub source")
            cache.write(fp, identity, data)
        return base64.b64decode(data["content"]).decode(), data["sha"]


def _parse_variant_consequences(ts: str) -> dict[str, str]:
    """Pull the ``VARIANT_CONSEQUENCES`` array out of variant.ts."""
    match = re.search(
        r"VARIANT_CONSEQUENCES\s*:\s*VariantConsequence\[\]\s*=\s*\[(.*?)\];",
        ts,
        re.DOTALL,
    )
    if match is None:
        raise RuntimeError("Could not find VARIANT_CONSEQUENCES in variant.ts")
    out: dict[str, str] = {}
    for entry in re.finditer(r'id:\s*"(SO:\d+)"\s*,\s*label:\s*"([^"]+)"', match.group(1)):
        # OT writes IDs as ``SO:0001583``; the OT *variant* parquet table
        # emits ``SO_0001583`` (underscore). Store the underscore form so
        # the renderer can do a direct dict lookup against the variant data.
        out[entry.group(1).replace(":", "_")] = entry.group(2)
    if not out:
        raise RuntimeError("VARIANT_CONSEQUENCES parsed to empty dict")
    return out


def _parse_variant_effect_methods(ts: str) -> dict[str, str]:
    """Pull ``VARIANT_EFFECT_METHODS`` (method -> prettyName) out of variant.ts."""
    match = re.search(
        r"VARIANT_EFFECT_METHODS\s*:\s*Record<string,\s*VariantEffectMethod>\s*=\s*\{(.*?)\n\};",
        ts,
        re.DOTALL,
    )
    if match is None:
        raise RuntimeError("Could not find VARIANT_EFFECT_METHODS in variant.ts")
    out: dict[str, str] = {}
    # Each entry has shape ``MethodKey: { methodName: "...", prettyName: "...", ... }``.
    for entry in re.finditer(
        r"(\w+):\s*\{\s*methodName:\s*\"([^\"]+)\",\s*prettyName:\s*\"([^\"]+)\"",
        match.group(1),
    ):
        # Use the TypeScript key (e.g. ``AlphaMissense``) as the lookup
        # because that's what OT's *data* emits in the
        # ``variantEffect[*].method`` field.
        out[entry.group(1)] = entry.group(3)
    if not out:
        raise RuntimeError("VARIANT_EFFECT_METHODS parsed to empty dict")
    return out


def _parse_population_labels(tsx: str) -> dict[str, str]:
    """Pull ``populationLabels`` (gnomAD code -> label) out of ProfileHeader.tsx."""
    match = re.search(
        r"const\s+populationLabels\s*=\s*\{(.*?)\};",
        tsx,
        re.DOTALL,
    )
    if match is None:
        raise RuntimeError("Could not find populationLabels in ProfileHeader.tsx")
    out: dict[str, str] = {}
    # OT formats this as ``key_adj: "Display Label",`` -- unquoted JS keys.
    for entry in re.finditer(
        r"(\w+_adj)\s*:\s*\"([^\"]+)\"",
        match.group(1),
    ):
        out[entry.group(1)] = entry.group(2)
    if not out:
        raise RuntimeError("populationLabels parsed to empty dict")
    return out


def _parse_str_to_int_map(ts: str, const_name: str) -> dict[str, int]:
    """Pull a ``Record<string, number>`` from index.ts (e.g. ``clinvarStarMap``)."""
    match = re.search(
        rf"export\s+const\s+{const_name}\s*:[^=]*=\s*\{{(.*?)\}};",
        ts,
        re.DOTALL,
    )
    if match is None:
        raise RuntimeError(f"Could not find {const_name} in index.ts")
    out: dict[str, int] = {}
    for entry in re.finditer(r'"([^"]+)"\s*:\s*(\d+)', match.group(1)):
        out[entry.group(1)] = int(entry.group(2))
    if not out:
        raise RuntimeError(f"{const_name} parsed to empty dict")
    return out


def fetch_constants(**kwargs: Any) -> dict[str, Any]:
    """Return parsed canonical maps and source SHAs from one acquisition pass."""
    variant, variant_sha = fetch_source(OT_VARIANT_TS, **kwargs)
    profile, profile_sha = fetch_source(OT_PROFILE_HEADER_TSX, **kwargs)
    index, index_sha = fetch_source(OT_INDEX_TS, **kwargs)
    return {
        "variant_consequences": _parse_variant_consequences(variant),
        "variant_effect_methods": _parse_variant_effect_methods(variant),
        "population_labels": _parse_population_labels(profile),
        "clinvar_star_map": _parse_str_to_int_map(index, "clinvarStarMap"),
        "credset_confidence_map": _parse_str_to_int_map(index, "credsetConfidenceMap"),
        "variant_ts_sha": variant_sha,
        "profile_header_sha": profile_sha,
        "index_ts_sha": index_sha,
    }


def fetch_sequence_ontology_labels(
    *, version: str = "25.12", cache_dir: str | Path | None = None
) -> dict[str, str]:
    """Resolve the complete SO snapshot through the public bulk acquisition API."""
    from biodb.opentargets import get_dataset

    frame = get_dataset("so", version=version, cache_dir=cache_dir, output_format="polars")
    if not {"id", "label"}.issubset(frame.columns):
        # OT releases have also named the label column ``name``.
        label = "name" if "name" in frame.columns else None
        if label is None or "id" not in frame.columns:
            raise ValueError(f"Unexpected SO schema: {frame.columns}")
    else:
        label = "label"
    return {str(row["id"]).replace(":", "_"): str(row[label]) for row in frame.to_dicts()}
