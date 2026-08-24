from ai_fingerprint.dataset_catalog import DATASETS, automatic_datasets


def test_expanded_catalog_has_at_least_thirty_datasets():
    assert len(DATASETS) >= 30


def test_manual_datasets_are_marked():
    assert DATASETS["imagenet"].acquisition == "manual"
    assert DATASETS["coco2017"].acquisition == "manual"


def test_automatic_download_filter_excludes_manual_and_large():
    names = automatic_datasets("small")
    assert "cifar10" in names
    assert "imagenet" not in names
    assert "amazon_polarity" not in names
