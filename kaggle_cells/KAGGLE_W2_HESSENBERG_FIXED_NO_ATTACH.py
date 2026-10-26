# INDEPENDENT KAGGLE CPU CELL — W2 — NO PRIOR ZIP OR DATA ATTACHMENT
# 1) Upload the five PATCH_GITHUB files to the SAME paths in GitHub, commit to main.
# 2) Kaggle: Internet=ON, Accelerator=None (CPU); paste this ONE cell.
import os, sys, json, subprocess, shutil, hashlib
from pathlib import Path

LABEL = "W2"
GIT = "https://github.com/tydeptrai21042004/pham_thi_yen.git"
WORK = Path("/kaggle/working")
assert WORK.is_dir(), "This must run in Kaggle"
REPO = WORK / ("yen_hess_fixed_" + LABEL.lower() + "_repo")
OUT = WORK / ("yen_" + LABEL.lower() + "_hessenberg_fixed")
ZIP = WORK / (OUT.name + ".zip")
os.environ.update(PYTHONUNBUFFERED="1", WATERMARKLAB_STRICT_PYWT="1",
                  OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  OMP_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                  MPLBACKEND="Agg")

if REPO.exists():
    shutil.rmtree(REPO)
subprocess.run(["git", "clone", "--depth", "1", "--branch", "main", GIT, str(REPO)], check=True)
COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
print("SOURCE COMMIT:", COMMIT, flush=True)
required = [
    "scripts/recover_missing_results.py",
    "scripts/run_remaining_fresh.py",
    "scripts/plot_remaining_fresh.py",
    "src/watermarklab/experiments/reviewer_checks.py",
    "tests/test_hessenberg_repair_runner.py",
]
missing = [name for name in required if not (REPO / name).is_file()]
if missing:
    raise RuntimeError("Commit the supplied GitHub patch first: " + str(missing))
# Enforce corrected runner and validator logic rather than silently using the old version.
assert "--focus-failed-baselines" in (REPO / required[1]).read_text(), "Old runner: patch missing"
assert "preset=args.reviewer_attack_preset" in (REPO / required[0]).read_text(), "Old validator: patch missing"
assert "            import hashlib\n" not in (REPO / required[0]).read_text(), "Old heldout hashlib bug"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=REPO, check=True)
env = os.environ.copy()
env["PYTHONPATH"] = str(REPO / "src")
subprocess.run([sys.executable, "-m", "pytest", "-q",
    "tests/test_hessenberg_repair_runner.py",
    "tests/test_recovery_resume_2026.py",
    "tests/test_undefined_nc_checkpoint_fix.py",
    "tests/test_full_reviewer_kaggle_contract_2026.py"], cwd=REPO, env=env, check=True)

# Fresh experiment: W1=wm.png, W2=alpha-derived Cc.logo.circle.png.
# 9 training hosts; 12 separate 512x512 test hosts; Lena excluded.
# Focus only failed classical baselines (Su, Nha, Gaata, Abodena, WHT),
# incomplete channel/BSS, and optional remaining manuscript extras.
# No TrustMark/WAM, no change to proposal method or Firefly objective.
command = [sys.executable, "scripts/run_remaining_fresh.py",
    "--label", LABEL,
    "--out", str(OUT),
    "--focus-failed-baselines",
    "--fireflies", "2", "--generations", "2",
    "--task-timeout", "10500",
    "--max-seconds", "40500"]
try:
    finished = subprocess.run(command, cwd=REPO, env=env, check=False)
    print("RUNNER EXIT CODE:", finished.returncode, flush=True)
finally:
    status = OUT / "fresh_remaining_status.json"
    if status.exists():
        result = json.loads(status.read_text())
        print("EXPERIMENT JOBS:", json.dumps(result.get("jobs", {}), indent=2), flush=True)
        print("ALL SCHEDULED JOBS COMPLETE:", result.get("all_scheduled_steps_passed"), flush=True)
    print("OUTPUT ZIP:", str(ZIP), "EXISTS:", ZIP.exists(), flush=True)
    if ZIP.exists():
        print("ZIP BYTES:", ZIP.stat().st_size, flush=True)
# A published partial checkpoint is not a certified complete manuscript result.
