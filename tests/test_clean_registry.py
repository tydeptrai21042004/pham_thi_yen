from watermarklab.methods import build_methods, BASELINE_METHOD_IDS, DEFAULT_METHOD_IDS


def test_clean_method_registry_contains_only_requested_methods():
    assert BASELINE_METHOD_IDS == [
        "kumar2021",
        "guo2017_dwt_qr_fa",
        "gaata2022_dwt_hess_fwa",
        "dwt_hd_svd_2025",
        "hess_nha2023",
        "roy2018_dwt_svd",
    ]
    assert DEFAULT_METHOD_IDS == BASELINE_METHOD_IDS + ["proposal"]
    assert list(build_methods(["baselines"]).keys()) == BASELINE_METHOD_IDS
    assert list(build_methods(["all"]).keys()) == DEFAULT_METHOD_IDS


def test_removed_methods_are_not_registered():
    methods = build_methods(["all"])
    assert "roy2018" not in methods
    assert "iwt_hess_svd_2024" not in methods
    assert "mahto2022_firefly_dual" not in methods
