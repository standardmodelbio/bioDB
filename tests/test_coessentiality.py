"""Tests for :mod:`biodb.coessentiality` on a tiny synthetic matrix."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import responses

from biodb import coessentiality as ce


def _npy_bytes(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.asfortranarray(array))
    return buf.getvalue()


@pytest.fixture
def matrices(tmp_path: Path) -> Path:
    genes = ["A", "B", "C", "D"]
    p = np.full((4, 4), 0.5)
    sign = np.ones((4, 4))
    p[0, 1] = p[1, 0] = 1e-9  # A-B strongly co-essential
    p[2, 3] = p[3, 2] = 1e-9
    sign[2, 3] = sign[3, 2] = -1.0  # ...but anti-correlated
    p[0, 2] = p[2, 0] = 0.02
    (tmp_path / "genes.txt").write_text("\n".join(genes) + "\n")
    (tmp_path / "GLS_p.npy").write_bytes(_npy_bytes(p))
    (tmp_path / "GLS_sign.npy").write_bytes(_npy_bytes(sign))
    return tmp_path


def test_bh_cutoff_is_the_largest_passing_p() -> None:
    p = np.array([0.001, 0.01, 0.03, 0.5, 0.9])
    # BH at 10%: thresholds 0.02, 0.04, 0.06, 0.08, 0.10 -> the first three pass.
    assert ce._bh_cutoff(p, 0.10) == 0.03
    assert ce._bh_cutoff(np.array([0.5, 0.9]), 0.10) == 0.0


def test_pairs_at_fdr_keep_positive_significant_pairs(matrices: Path) -> None:
    # Six upper-triangle p values: 1e-9, 1e-9, 0.02 and three 0.5. BH at 1 %
    # passes only the 1e-9 pairs; at 10 % the 0.02 pair (rank 3, threshold
    # 0.05) passes too.
    net = ce.coessential_pairs(fdr=0.01, cache_dir=matrices, progress=False)
    assert net.select("gene_a", "gene_b").rows() == [("A", "B")]
    ten = ce.coessential_pairs(fdr=0.10, cache_dir=matrices, progress=False)
    assert ten.select("gene_a", "gene_b").rows() == [("A", "B"), ("A", "C")]
    both = ce.coessential_pairs(fdr=0.01, positive_only=False, cache_dir=matrices, progress=False)
    assert both.select("gene_a", "gene_b").rows() == [("A", "B"), ("C", "D")]
    loose = ce.coessential_pairs(p_threshold=0.05, cache_dir=matrices, progress=False)
    assert loose.select("gene_a", "gene_b").rows() == [("A", "B"), ("A", "C")]
    subset = ce.coessential_pairs(
        p_threshold=0.05, genes=["a", "c"], cache_dir=matrices, progress=False
    )
    assert subset.select("gene_a", "gene_b").rows() == [("A", "C")]
    with pytest.raises(ValueError, match="not both"):
        ce.coessential_pairs(fdr=0.1, p_threshold=0.1, cache_dir=matrices)


def test_pair_lookup_returns_null_for_unknown_genes(matrices: Path) -> None:
    got = ce.pair_p_values(
        [("A", "B"), ("b", "a"), ("A", "ZZZ"), ("A", "A")], cache_dir=matrices, progress=False
    )
    assert got["p"].to_list() == pytest.approx([1e-9, 1e-9, None, None])
    assert got["sign"].to_list()[:2] == [1.0, 1.0]


@responses.activate
def test_download_fetches_the_three_files_once(tmp_path: Path) -> None:
    for name, body in (
        ("genes.txt", b"A\nB\n"),
        ("GLS_p.npy", _npy_bytes(np.full((2, 2), 0.5))),
        ("GLS_sign.npy", _npy_bytes(np.ones((2, 2)))),
    ):
        responses.add(responses.GET, f"{ce.BASE_URL}/{name}", body=body, status=200)
    paths = ce.download_matrices(cache_dir=tmp_path, progress=False)
    assert set(paths) == set(ce.FILES) and all(p.exists() for p in paths.values())
    ce.download_matrices(cache_dir=tmp_path, progress=False)
    assert len(responses.calls) == 3
    assert ce.load_genes(cache_dir=tmp_path, progress=False) == ["A", "B"]
