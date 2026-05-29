from watermarklab.benchmark import run_benchmark

if __name__ == "__main__":
    run_benchmark(
        host_dir="data/host",
        watermark_path="data/watermark/wm.png",
        output_dir="results/common_benchmark",
        selected_methods=["all"],
        max_images=None,
        save_outputs=True,
        attack_preset="lite",
        proposal_options={"params": {"repeat": 3}, "use_optimizer": False},
    )
    print("Done. See results/common_benchmark/compare_psnr_nc_ber_ncc_before_after_attack.csv")
