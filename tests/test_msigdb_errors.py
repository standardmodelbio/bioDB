"""Preserve HTTP failures while recognizing MSigDB's explicit missing-set route."""

import pytest
import requests
import responses

from biodb import msigdb


@pytest.mark.parametrize("status", [200, 400, 404])
def test_missing_set_redirect_is_a_key_error(status):
    url = msigdb.MSIGDB_GENESET_URL_TEMPLATE.format(organism="human", set_name="MISSING")
    missing = "https://www.gsea-msigdb.org/gsea/msigdb/no_such_gene_set.jsp"
    with responses.RequestsMock() as mock:
        mock.get(url, status=302, headers={"Location": missing})
        mock.get(missing, status=status, body="No such gene set", content_type="text/html")
        with pytest.raises(KeyError, match="not found"):
            msigdb.query_gene_set("MISSING")


@pytest.mark.parametrize("status", [400, 403, 429, 500, 503])
def test_other_http_errors_are_not_missing_sets(status):
    url = msigdb.MSIGDB_GENESET_URL_TEMPLATE.format(organism="human", set_name="KNOWN")
    with responses.RequestsMock() as mock:
        mock.get(url, status=status)
        with pytest.raises(requests.HTTPError) as caught:
            msigdb.query_gene_set("KNOWN")
        assert caught.value.response.status_code == status


def test_missing_set_route_with_server_error_still_fails():
    url = msigdb.MSIGDB_GENESET_URL_TEMPLATE.format(organism="human", set_name="MISSING")
    missing = "https://www.gsea-msigdb.org/gsea/msigdb/no_such_gene_set.jsp"
    with responses.RequestsMock() as mock:
        mock.get(url, status=302, headers={"Location": missing})
        mock.get(missing, status=503)
        with pytest.raises(requests.HTTPError):
            msigdb.query_gene_set("MISSING")
