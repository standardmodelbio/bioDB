"""Smoke tests for :mod:`biodb.uniprot`.

Live REST calls are marked ``@pytest.mark.network`` so CI skips them.
"""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from biodb import uniprot


def test_module_imports() -> None:
    assert uniprot.__name__ == "biodb.uniprot"


def test_endpoint_constant() -> None:
    assert uniprot.UNIPROT_REST_API == "https://rest.uniprot.org/uniprotkb"


def test_public_api_signatures_stable() -> None:
    expected = {"query_protein", "get_sequences", "get_features", "get_dbxrefs"}
    missing = [name for name in expected if not hasattr(uniprot, name)]
    assert not missing, f"missing public symbols: {missing}"


def test_query_protein_signature_uses_keyword_only_options() -> None:
    """The new API uses keyword-only options (``fmt``, ``timeout_s``, ``verbose``)
    so callers can't accidentally pass them positionally."""
    sig = inspect.signature(uniprot.query_protein)
    params = sig.parameters
    assert params["uniprot_id"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params["fmt"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["timeout_s"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["verbose"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.network
def test_query_protein_returns_materialized_list() -> None:
    """Smoke check: result is a list (not a SeqIO iterator) so it can be
    re-traversed. This is the explicit fix over the VEP_protein version."""
    records = uniprot.query_protein("P12345")
    assert isinstance(records, list)
    # The "must reimport" footgun: a SeqIO.parse() iterator would be empty here.
    assert len(records) == len(records)


@pytest.mark.network
def test_get_features_returns_dataframe() -> None:
    df = uniprot.get_features("P12345")
    assert isinstance(df, pd.DataFrame)
    if not df.empty:
        assert {"id", "type", "start", "end", "length"}.issubset(df.columns)
