# Single-firefly Kaggle launcher correction

Overlay the `pham_thi_yen-main/` contents on the existing GitHub repo root, then commit and push. Run the revised `KAGGLE_GITHUB_ALL_EXPERIMENTS.py` as one Kaggle notebook cell. The patch contains only the launcher and tests. It **does not modify the proposal method source**.

- Proposal Firefly **1 individual, 2 generations**; mathematically this is a *single initial candidate*, since it has no other firefly to move toward. It is not a multi-individual Firefly search; label it accurately in the paper.
- Enable both 4,096-bit TrustMark and WAM *adaptations*; full attacks forced, failures measured rather than suppressed.
- Skip native 61-bit TrustMark and native 32-bit WAM.
- Run full classical benchmark, reviewer experiments, sensitivity, gallery, separate second watermark.
- Session soft budget 31,500 s (8h45m) with periodic checkpoints; use returned ZIP as Kaggle Input to resume on a new session.
- Existing checkpoints from 4-firefly optimization will be rejected by optimizer configuration validation.

This patch fixes the premature assertion and test contract. It does **not** guarantee that all long computational experiments finish in one Kaggle session.
