from pathlib import Path

from watermarklab.benchmark import run_proposal_ablation_phase


def test_proposal_ablation_phase_smoke(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "proposal_ablation"

    result = run_proposal_ablation_phase(
        host_dir=root / "data/host",
        watermark_path=root / "data/watermark/wm.png",
        output_dir=out,
        variants="full,no_structured_repetition",
        max_images=1,
        save_outputs=False,
        attack_preset="none",
    )

    assert not result["results"].empty
    assert set(result["results"]["proposal_ablation"]) == {"full", "no_structured_repetition"}
    assert (out / "proposal_ablation_manifest.csv").exists()
    assert (out / "proposal_ablation_per_image_attack_results.csv").exists()
    assert (out / "proposal_ablation_summary_by_variant_phase.csv").exists()
    assert (out / "proposal_ablation_delta_vs_full.csv").exists()
    assert (out / "proposal_ablation_failures.json").exists()
