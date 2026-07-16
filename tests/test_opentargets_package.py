"""Back-compat + package-structure tests for the opentargets package."""


def test_bulk_submodule_exists():
    # Fails before the move: there is no _bulk submodule yet.
    from biodb.opentargets import _bulk

    assert hasattr(_bulk, "get_dataset")


def test_public_names_still_importable_from_package():
    # The historical flat-module import surface must keep working.
    from biodb.opentargets import (  # noqa: F401
        DEFAULT_VERSION,
        ensure_cached_shards,
        get_dataset,
        get_gene_associations,
        get_pathways,
        get_targets,
        list_available_versions,
        list_datasets,
        read_for_target,
        variants_for_target,
    )

    assert isinstance(DEFAULT_VERSION, str)


def test_reexports_are_identical_objects():
    import biodb.opentargets as ot
    from biodb.opentargets import _bulk

    assert ot.get_dataset is _bulk.get_dataset
    assert ot.DEFAULT_VERSION == _bulk.DEFAULT_VERSION


def test_module_style_access_still_works():
    from biodb import opentargets

    assert callable(opentargets.get_dataset)
