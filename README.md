# PyramidHBM Illustrative Example

This script reproduces the toy example from the paper "Pyramid-Based Bayesian Modeling for High-Resolution Behavioral Analysis".

## Key features demonstrated:
- **PyramidHBMc**: covariance matrix **only** at the coarsest level (`n_coarse=2`) + hierarchical variance-only refinement below
- Comparison to: BIP (independent), FullHBMc (full covariance everywhere)
- Three cases: matching assumptions, violating assumptions, extreme sparsity
- Metrics: posterior mean RMSE + runtime (WAIC removed for simplicity)

## Requirements
- PyMC, ArviZ, NumPy, Matplotlib, Pandas

Run the script directly — it will generate three PDF plots (`illustrative_matching.pdf`, etc.) and print RMSE/runtime tables.
