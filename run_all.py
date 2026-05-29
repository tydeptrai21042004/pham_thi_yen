from watermarklab.benchmark import run_benchmark, load_proposal_param_file

if __name__ == "__main__":
    optimized = load_proposal_param_file("results/proposal_optimized_params.json")
    run_benchmark(
        host_dir="data/host",
        watermark_path="data/watermark/wm.png",
        output_dir="results/common_benchmark",
        selected_methods=["all"],
        max_images=None,
        save_outputs=True,
        attack_preset="lite",
        proposal_options={
            "params": {"repeat": None, "dwt_mode": "pywt"},
            "use_optimizer": False,
            "optimized_payload": optimized if optimized.get("loaded") else None,
        },
    )
    print("Done. See results/common_benchmark/compare_psnr_nc_ber_ncc_before_after_attack.csv")
