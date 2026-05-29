from pathlib import Path

from watermarklab.benchmark import run_benchmark


def test_benchmark_generates_expected_csvs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "run"
    result = run_benchmark(
        host_dir=root / "data/host",
        watermark_path=root / "data/watermark/wm.png",
        output_dir=out,
        selected_methods=["kumar2021", "guo2017_dwt_qr_fa"],
        max_images=1,
        save_outputs=False,
        attack_preset="none",
    )
    assert not result["results"].empty
    assert (out / "per_image_attack_results.csv").exists()
    assert (out / "summary_by_method_phase.csv").exists()
    assert (out / "compare_psnr_nc_ber_ncc_before_after_attack.csv").exists()
    assert (out / "failures.json").exists()
