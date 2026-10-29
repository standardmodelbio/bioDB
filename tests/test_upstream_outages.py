"""Live-test availability guards must never hide changed API contracts."""

import pytest
import requests

from tests.conftest import skip_upstream_outages


def http_error(status):
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(response=response)


@pytest.mark.parametrize(
    "error",
    [requests.ConnectionError(), requests.Timeout(), *[http_error(s) for s in (502, 503, 504)]],
)
def test_unavailable_upstream_is_reported_as_skip(error):
    with (
        pytest.raises(pytest.skip.Exception, match="GWAS Atlas upstream outage"),
        skip_upstream_outages("GWAS Atlas"),
    ):
        raise error


@pytest.mark.parametrize(
    "error",
    [*[http_error(s) for s in (400, 403, 404, 429, 500)], ValueError("bad JSON"), AssertionError()],
)
def test_contract_and_parse_failures_still_fail(error):
    with pytest.raises(type(error)) as caught, skip_upstream_outages("GWAS Atlas"):
        raise error
    assert caught.value is error
