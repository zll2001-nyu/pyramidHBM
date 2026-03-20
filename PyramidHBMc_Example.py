import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import warnings

# Suppress non-critical warnings from ArviZ/PyMC to keep output clean
warnings.filterwarnings("ignore", category=UserWarning, module="arviz")

# Set random seed for reproducibility
np.random.seed(42)

# ========================= CONFIGURATION =========================
# Number of subjects (individuals)
n_subjects = 20

# Total number of blocks (finest resolution; must be power of 2 for halving pyramid)
n_blocks = 8

# Number of coarse blocks at the top layer (2^1 = 2 here → halves twice to reach 8)
n_coarse = 2

# (Unused in this version but kept for polynomial variant reference)
n_poly_order = 2


# ========================= DATA GENERATION =========================
def generate_data(case="matching", sparse=False):
    """
    Generate synthetic data for the illustrative example.
    
    Parameters:
        case: str
            "matching"  → mild fine-scale noise + shared mild offsets (pyramid assumptions hold well)
            "violating" → stronger fine-scale correlations that violate pyramid smoothness
            "sparse"    → higher observation noise to simulate extreme sparsity
        sparse: bool
            If True, increases observation noise (sigma = 0.28 vs 0.10)
    
    Returns:
        true_theta: (n_blocks,) array of ground-truth latent parameters
        y_obs:      (n_subjects, n_blocks) observed data
    """
    # True coarse values (repeated across 4 blocks each)
    coarse_true = np.array([0.3, 0.7])
    
    # Observation noise level (higher = sparser / harder)
    obs_sigma = 0.28 if sparse else 0.10
    
    if case == "matching":
        # Mild independent fine noise + mild shared structure (good for pyramid)
        fine_noise = np.random.normal(0, 0.05, n_blocks)
        true_theta = np.repeat(coarse_true, 4) + fine_noise
        shared_mild = np.random.normal(0, 0.08, 4)
        true_theta += np.repeat(shared_mild, 2)
    else:  # "violating"
        # Stronger fine noise + sinusoidal correlation in second half (challenges pyramid)
        fine_noise = np.random.normal(0, 0.08, n_blocks)
        true_theta = np.repeat(coarse_true, 4) + fine_noise
        true_theta[4:] += 0.35 * np.sin(np.linspace(0, 4*np.pi, 4))
        shared_corr = np.random.normal(0, 0.15, 4)
        true_theta[4:] += np.repeat(shared_corr, 1)
    
    # Subject-specific random offsets
    subject_offset = np.random.normal(0, 0.08, n_subjects)
    
    # Generate observations
    y = np.zeros((n_subjects, n_blocks))
    for s in range(n_subjects):
        y[s] = true_theta + subject_offset[s] + np.random.normal(0, obs_sigma, n_blocks)
    
    return true_theta, y


# ========================= MODEL DEFINITIONS =========================
# All models use weakly informative Normal priors on means and Exponential on SDs

def build_bip_model(y_obs):
    """Baseline: fully independent parameters per subject/block (no hierarchy)."""
    with pm.Model() as m:
        theta = pm.Normal('theta', mu=0, sigma=1, shape=(n_subjects, n_blocks))
        sigma_obs = pm.Exponential('sigma_obs', lam=1.0)
        pm.Normal('obs', mu=theta, sigma=sigma_obs, observed=y_obs)
    return m


def build_full_hbmc_model(y_obs):
    """Full covariance HBMc across ALL blocks → suffers from large covariance matrix."""
    with pm.Model() as m:
        mu_pop = pm.Normal('mu_pop', mu=0, sigma=1, shape=n_blocks)
        sd_dist = pm.Exponential.dist(lam=2.0, shape=n_blocks)
        chol, _, _ = pm.LKJCholeskyCov('chol', eta=2, n=n_blocks, sd_dist=sd_dist)
        theta = pm.MvNormal('theta', mu=mu_pop, chol=chol, shape=(n_subjects, n_blocks))
        sigma_obs = pm.Exponential('sigma_obs', lam=1.0)
        pm.Normal('obs', mu=theta, sigma=sigma_obs, observed=y_obs)
    return m


def build_pyramid_hbmc_model(y_obs):
    """
    Proposed PyramidHBMc model:
      - Top layer: full covariance (HBMc) ONLY on the coarsest scale (n_coarse blocks)
      - Lower layers: variance-only hierarchical refinement (HBMv style) on differences
    """
    with pm.Model() as m:
        # === Top layer: coarse scale with full covariance (this is the only place covariance appears) ===
        mu_coarse = pm.Normal('mu_coarse', mu=0, sigma=1, shape=n_coarse)
        sd_dist_coarse = pm.Exponential.dist(lam=2.0, shape=n_coarse)
        chol_coarse, _, _ = pm.LKJCholeskyCov('chol_coarse', eta=2, n=n_coarse, sd_dist=sd_dist_coarse)
        theta_coarse = pm.MvNormal('theta_coarse', mu=mu_coarse, chol=chol_coarse,
                                    shape=(n_subjects, n_coarse))

        # === Difference pyramid layers: hierarchical variance-only (HBMv) refinement ===
        theta_recon = theta_coarse[:, 0]  # Start with coarsest level (repeated later)
        for layer in range(1, int(np.log2(n_blocks)) + 1):
            n_this = n_coarse * (2 ** layer)  # resolution doubles each layer
            mu_diff = pm.Normal(f'mu_diff_l{layer}', mu=0, sigma=1, shape=n_this)
            sd_diff = pm.Exponential(f'sd_diff_l{layer}', lam=1.0, shape=n_this)
            diff = pm.Normal(f'diff_l{layer}', mu=mu_diff, sigma=sd_diff,
                             shape=(n_subjects, n_this))
            # Cumulative reconstruction: add refinement to previous level
            theta_recon = pm.Deterministic(f'theta_recon_l{layer}', theta_recon + diff)

        # Observation model
        sigma_obs = pm.Exponential('sigma_obs', lam=1.0)
        pm.Normal('obs', mu=theta_recon, sigma=sigma_obs, observed=y_obs)
    return m


# ========================= MODEL FITTING HELPER =========================
def fit_model(model_name, y_obs, true_theta):
    """
    Fit a single model using ADVI + sampling, compute RMSE of posterior mean.
    
    Returns: idata, rmse, runtime
    """
    start = time.time()
    
    if model_name == 'BIP':
        model = build_bip_model(y_obs)
    elif model_name == 'FullHBMc':
        model = build_full_hbmc_model(y_obs)
    elif model_name == 'PyramidHBMc':
        model = build_pyramid_hbmc_model(y_obs)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    with model:
        # Use ADVI for fast approximate inference (good for illustration)
        approx = pm.fit(method='advi', n=50000, progressbar=False)
        # Sample posterior
        idata = approx.sample(draws=2000)
    
    runtime = time.time() - start
    
    # Extract posterior mean of reconstructed theta
    var_name = 'theta_recon' if model_name == 'PyramidHBMc' else 'theta'
    summary = az.summary(idata, var_names=[var_name])
    post_mean = summary['mean'].values.reshape(n_subjects, n_blocks).mean(axis=0)
    rmse = np.sqrt(np.mean((post_mean - true_theta)**2))
    
    print(f"{model_name} | RMSE: {rmse:.4f} | Time: {runtime/60:.1f} min")
    return idata, rmse, runtime


# ========================= MAIN EXPERIMENT LOOP =========================
cases = ["matching", "violating", "sparse"]

for case in cases:
    print(f"\n{'='*70}\nCASE: {case.upper()}\n{'='*70}")
    
    # Generate data for this case
    true_theta, y_obs = generate_data(case, sparse=(case == "sparse"))
    
    # Models to compare
    models = ['BIP', 'FullHBMc', 'PyramidHBMc']
    
    results = {}
    for m in models:
        idata, rmse, rt = fit_model(m, y_obs, true_theta)
        results[m] = {'rmse': rmse, 'time': rt, 'idata': idata}
    
    # Summary table (RMSE + runtime only)
    df = pd.DataFrame(
        {m: [results[m]['rmse'], results[m]['time']/60] for m in models},
        index=['RMSE', 'Time (min)']
    )
    print("\nModel Comparison:")
    print(df.round(3))
    
    # Plot posterior means vs true theta
    fig, axs = plt.subplots(1, len(models), figsize=(4*len(models), 5), sharey=True)
    x = np.arange(n_blocks)
    for i, m in enumerate(models):
        var_name = 'theta_recon' if m == 'PyramidHBMc' else 'theta'
        summary = az.summary(results[m]['idata'], var_names=[var_name])
        post_mean = summary['mean'].values.reshape(n_subjects, n_blocks).mean(0)
        axs[i].plot(x, true_theta, 'k-', lw=2.5, label='True θ')
        axs[i].plot(x, post_mean, 'o-', label=m, markersize=5)
        axs[i].set_title(m)
        axs[i].legend()
        axs[i].set_xlabel('Block index')
        axs[i].set_ylim(0, 1.4)
    axs[0].set_ylabel('Latent parameter θ')
    plt.tight_layout()
    plt.savefig(f'illustrative_{case}.pdf', dpi=300, bbox_inches='tight')
    plt.show()

print("\n" + "="*70)
print("Finished. All cases complete.")
print("PyramidHBMc uses full covariance ONLY at the top (coarse) layer,")
print("with variance-only hierarchical refinement (HBMv style) in difference layers.")
print("Results saved as PDFs in current directory.")
