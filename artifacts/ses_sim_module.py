
"""
SES Simulation Scaffold (Python)
================================

A comprehensive simulation framework for studying socioeconomic status (SES) effects on health
outcomes through complex, non-linear causal pathways with selection bias and measurement error.

Overview
--------
This module implements a sophisticated data generating mechanism that simulates realistic
relationships between socioeconomic status and disease outcomes, incorporating:

1. **Non-linear SES effects**: Complex relationships between true SES and health mediators
2. **Selection bias**: Biobank-like participation patterns that create systematic sampling bias
3. **Measurement error**: Noisy observation of true SES, mimicking real-world data limitations
4. **Multiple analytical approaches**: Comparison of traditional vs. modern causal inference methods

Key Features
------------
- **Realistic Data Generation**: Simulates population-level data with complex causal relationships
- **Selection Mechanisms**: Models biobank participation bias based on health and socioeconomic factors
- **Multiple SES Representations**: True latent SES*, observed noisy SES, and categorical quintiles
- **Flexible Modeling**: Supports linear models, splines, and g-computation for causal inference
- **Performance Metrics**: Multiple R² variants and Shapley value decomposition for model comparison

Analytical Approaches Compared
------------------------------
A) **Traditional Mis-specified Models**:
   - Linear SES effects using categorical quintiles
   - Over-adjusted models including mediators (collider bias)

B) **Flexible Spline Models**:
   - Non-parametric modeling of SES-outcome relationships
   - Comparison between oracle (SES*) and proxy (SES_obs) versions

C) **G-computation Dose-Response**:
   - Causal intervention curves: E[Y | do(SES = a)]
   - Oracle estimates using true SES* vs. proxy estimates using observed SES

Mathematical Framework
----------------------
The data generating process follows this causal structure:
    C (confounders) → SES* → M (mediators) → Y (outcome)
                  ↓                        ↓
               SES_obs                      S (selection)

Where:
- C: Baseline confounders (age, sex, genetic PCs)
- SES*: True latent socioeconomic status
- SES_obs: Observed SES proxy with measurement error
- M: Health-related mediators (BMI, smoking, blood pressure)
- Y: Binary disease outcome
- S: Biobank participation (selection indicator)

Usage Example
-------------
>>> # Basic simulation run
>>> results, pop, biobank, *curves = run_demo(n_pop=50000, seed=123)
>>> print(f"Population size: {results['n_population']}")
>>> print(f"Biobank sample: {results['n_biobank']}")
>>> print(f"True causal R²: {results['causal_R2_true']:.3f}")

>>> # Custom parameters
>>> params = DGMParams(n=100000, seed=456, ses_me_sd=0.5)
>>> pop_data = simulate_population(params)
>>> biobank_data = sample_biobank(pop_data)

Dependencies
------------
- numpy: Numerical computations and random number generation
- pandas: Data manipulation and analysis
- statsmodels: Statistical modeling and GLM fitting
- patsy: Formula interface for statistical models (splines)

Author
------
Dr. Ross A. Dunne

Notes
-----
This simulation is designed for methodological research comparing different approaches
to estimating socioeconomic effects on health outcomes in the presence of selection bias
and measurement error. The default parameters are calibrated to produce realistic effect
sizes and prevalence rates typical of epidemiological studies.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix

# -----------------
# Helper functions
# -----------------

def logistic(x: np.ndarray) -> np.ndarray:
    """
    Compute the logistic (sigmoid) function.
    
    Transforms real-valued inputs to probabilities in (0, 1) using the logistic function:
    σ(x) = 1 / (1 + exp(-x))
    
    Parameters
    ----------
    x : array-like
        Input values to transform. Can be scalar, vector, or array.
    
    Returns
    -------
    np.ndarray
        Logistic transformation of input, bounded in (0, 1).
        
    Examples
    --------
    >>> logistic(0)
    0.5
    >>> logistic(np.array([-2, 0, 2]))
    array([0.119, 0.5, 0.881])
    """
    return 1.0 / (1.0 + np.exp(-x))

def standardize(x: np.ndarray) -> np.ndarray:
    """
    Standardize a variable to have mean 0 and standard deviation 1.
    
    Applies z-score normalization: z = (x - μ) / σ
    
    Parameters
    ----------
    x : array-like
        Input variable to standardize.
    
    Returns
    -------
    np.ndarray
        Standardized variable with mean ≈ 0 and std ≈ 1.
        
    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> z = standardize(x)
    >>> np.mean(z), np.std(z)
    (0.0, 1.0)
    """
    return (x - np.mean(x)) / np.std(x)

def mc_fadden_r2(model_llf: float, null_llf: float) -> float:
    """
    Calculate McFadden's pseudo-R² for logistic regression models.
    
    McFadden's R² is a measure of goodness-of-fit for maximum likelihood models,
    analogous to R² in linear regression. It compares the log-likelihood of the
    fitted model to that of a null (intercept-only) model.
    
    Formula: R²_McF = 1 - (LL_model / LL_null)
    
    Parameters
    ----------
    model_llf : float
        Log-likelihood of the fitted model.
    null_llf : float
        Log-likelihood of the null (intercept-only) model.
    
    Returns
    -------
    float
        McFadden's pseudo-R², typically between 0 and 1.
        Values of 0.2-0.4 are considered excellent fit.
        
    Notes
    -----
    Unlike linear R², McFadden's R² does not represent proportion of variance
    explained. It measures relative improvement in log-likelihood over the null model.
    """
    return 1 - (model_llf / null_llf)

def tjur_r2(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    """
    Calculate Tjur's coefficient of discrimination (R²_D) for binary outcomes.
    
    Tjur's R² measures the difference in mean predicted probabilities between
    cases (y=1) and controls (y=0). It provides an intuitive measure of how well
    the model discriminates between the two groups.
    
    Formula: R²_D = E[p̂|Y=1] - E[p̂|Y=0]
    
    Parameters
    ----------
    y_true : array-like
        True binary outcomes (0 or 1).
    p_hat : array-like
        Predicted probabilities from the model.
    
    Returns
    -------
    float
        Tjur's R², ranging from 0 (no discrimination) to 1 (perfect discrimination).
        
    Notes
    -----
    Tjur's R² is bounded between 0 and 1 and has a direct interpretation:
    it represents the difference in average predicted probability between
    positive and negative cases.
    
    References
    ----------
    Tjur, T. (2009). Coefficients of determination in logistic regression models.
    The American Statistician, 63(4), 366-372.
    """
    y_true = np.asarray(y_true)
    p_hat = np.asarray(p_hat)
    return float(np.mean(p_hat[y_true==1]) - np.mean(p_hat[y_true==0]))

def partial_r2_mcfadden(llf_full: float, llf_reduced: float, null_llf: float) -> float:
    """
    Calculate partial (incremental) McFadden's R² for nested model comparison.
    
    Measures the additional explanatory power gained by adding variables to a model,
    expressed as the difference in McFadden's R² between the full and reduced models.
    
    Formula: ΔR²_McF = R²_McF(full) - R²_McF(reduced)
    
    Parameters
    ----------
    llf_full : float
        Log-likelihood of the full model (with additional variables).
    llf_reduced : float
        Log-likelihood of the reduced model (baseline).
    null_llf : float
        Log-likelihood of the null (intercept-only) model.
    
    Returns
    -------
    float
        Incremental McFadden's R² contributed by the additional variables.
        Positive values indicate improvement in model fit.
        
    Notes
    -----
    This measure is useful for assessing the contribution of specific variable groups
    in nested model comparisons, similar to partial R² in linear regression.
    """
    return mc_fadden_r2(llf_full, null_llf) - mc_fadden_r2(llf_reduced, null_llf)

def shapley_two_groups_r2(llf_null: float, llf_g1: float, llf_g2: float, llf_both: float) -> Dict[str, float]:
    """
    Compute Shapley value decomposition of McFadden's R² for two covariate groups.
    
    Applies game-theoretic Shapley values to fairly attribute the total model R²
    between two groups of covariates, accounting for their interactions and
    order-dependence in contribution calculations.
    
    The Shapley value represents each group's average marginal contribution across
    all possible orders of adding groups to the model, providing a fair allocation
    of the total explanatory power.
    
    Parameters
    ----------
    llf_null : float
        Log-likelihood of the null (intercept-only) model.
    llf_g1 : float
        Log-likelihood of model with group 1 covariates only.
    llf_g2 : float  
        Log-likelihood of model with group 2 covariates only.
    llf_both : float
        Log-likelihood of full model with both groups.
    
    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'G1_share': Shapley value for group 1 (average marginal contribution)
        - 'G2_share': Shapley value for group 2 (average marginal contribution)  
        - 'R2_total': Total McFadden's R² of the full model
        
    Notes
    -----
    For two groups, the Shapley value is the average of marginal contributions
    when added first vs. second:
    
    \u03c6₁ = 0.5 × [(R²₁ - R²₀) + (R²₁₂ - R²₂)]
    \u03c6₂ = 0.5 × [(R²₂ - R²₀) + (R²₁₂ - R²₁)]
    
    Where R²ᵢ denotes McFadden's R² for the model with group i.
    
    Examples
    --------
    >>> # Example with SES and demographic covariates
    >>> shap_results = shapley_two_groups_r2(
    ...     llf_null=-1000, llf_g1=-950, llf_g2=-980, llf_both=-920
    ... )
    >>> print(f"SES contribution: {shap_results['G1_share']:.3f}")
    >>> print(f"Demographics contribution: {shap_results['G2_share']:.3f}")
    """
    r2_null = 0.0
    r2_g1 = 1 - (llf_g1 / llf_null)
    r2_g2 = 1 - (llf_g2 / llf_null)
    r2_both = 1 - (llf_both / llf_null)

    # marginal contributions when added first vs second
    contrib_g1_first = r2_g1 - r2_null
    contrib_g1_second = r2_both - r2_g2
    contrib_g2_first = r2_g2 - r2_null
    contrib_g2_second = r2_both - r2_g1

    return {
        "G1_share": float(0.5 * (contrib_g1_first + contrib_g1_second)),
        "G2_share": float(0.5 * (contrib_g2_first + contrib_g2_second)),
        "R2_total": float(r2_both),
    }

# -----------------
# Data Generating Mechanism
# -----------------

@dataclass
class DGMParams:
    """
    Parameters for the Data Generating Mechanism (DGM) in SES simulation.
    
    This class encapsulates all tunable parameters that control the simulation of
    population data, including sample size, selection bias mechanisms, measurement
    error specifications, and outcome prevalence calibration.
    
    Attributes
    ----------
    n : int, default=200_000
        Total population size for simulation. Larger values provide more stable
        estimates but increase computational time. Recommended: 50K-500K.
        
    seed : int, default=42
        Random seed for reproducible results across simulation runs.
        
    Selection Bias Parameters
    -------------------------
    These parameters control biobank participation probability via logistic model:
    logit(P(S=1)) = sel_intercept + sel_ses*SES* + sel_smoke*smoke + ...
    
    sel_intercept : float, default=-2.2
        Intercept term controlling overall participation rate. More negative values
        lead to lower participation rates. Default yields ~5-15% participation.
        
    sel_ses : float, default=0.6
        Effect of true SES on participation. Positive values mean higher SES
        individuals are more likely to participate (typical in biobanks).
        
    sel_smoke : float, default=-0.8
        Effect of smoking on participation. Negative values reflect that smokers
        are less likely to participate in health studies.
        
    sel_bmi : float, default=-0.08
        Effect of BMI (per unit above 25) on participation. Negative values
        indicate that higher BMI reduces participation likelihood.
        
    sel_age : float, default=0.10
        Effect of standardized age on participation. Positive values reflect
        that older individuals may be more likely to participate.
        
    sel_risk_score : float, default=-0.7
        Effect of overall health risk score on participation. Negative values
        implement "healthy volunteer bias" - healthier individuals more likely
        to participate.
        
    Measurement Error Parameters
    ----------------------------
    ses_me_sd : float, default=0.7
        Standard deviation of measurement error in observed SES proxy.
        Higher values create more attenuation bias. Typical range: 0.3-1.0.
        Formula: SES_obs = 0.8*SES* + N(0, ses_me_sd²)
        
    Outcome Parameters
    ------------------
    y_intercept : float, default=-4.1
        Intercept term in outcome logistic model controlling baseline disease
        prevalence. Adjust to achieve desired population prevalence (target: 10-20%).
        More negative values reduce prevalence.
        
    Notes
    -----
    Default parameters are calibrated to produce realistic epidemiological patterns:
    - Participation rates similar to large biobanks (5-15%)
    - Healthy volunteer bias and socioeconomic selection
    - Moderate measurement error in SES proxy
    - Disease prevalence typical of chronic conditions
    
    Examples
    --------
    >>> # Default parameters
    >>> params = DGMParams()
    >>> 
    >>> # High measurement error scenario
    >>> params_noisy = DGMParams(ses_me_sd=1.2, n=100_000)
    >>> 
    >>> # Reduced selection bias
    >>> params_less_bias = DGMParams(sel_ses=0.2, sel_risk_score=-0.3)
    """
    n: int = 200_000
    seed: int = 42
    
    # Selection strength parameters
    sel_intercept: float = -2.2   # controls sampling fraction (~5-15% typical for demo)
    sel_ses: float = 0.6
    sel_smoke: float = -0.8
    sel_bmi: float = -0.08  # per unit BMI above 25
    sel_age: float = 0.10
    sel_risk_score: float = -0.7  # healthier more likely to volunteer

    # Measurement error for observed SES proxy
    ses_me_sd: float = 0.7

    # Outcome base rate calibrator
    y_intercept: float = -4.1  # tweak to get ~10-20% prevalence in population

def simulate_population(params: DGMParams) -> pd.DataFrame:
    """
    Generate synthetic population data with complex SES-health relationships.
    
    Creates a realistic population dataset incorporating non-linear relationships
    between socioeconomic status and health outcomes, with multiple mediating
    pathways and selection mechanisms that mirror real-world biobank studies.
    
    The data generating process follows this causal structure:
    C (age, sex, PC1) → SES* → M (BMI, smoking, SBP) → Y (disease)
                     ↓                              ↓
                  SES_obs                            S (selection)
    
    Parameters
    ----------
    params : DGMParams
        Configuration object containing all simulation parameters including
        sample size, selection mechanisms, and measurement error specifications.
        
    Returns
    -------
    pd.DataFrame
        Simulated population data with columns:
        - age: Age in years (40-85, mean≈60)
        - sex: Binary sex indicator (0=female, 1=male)
        - pc1: First principal component (genetic/ancestry proxy)
        - ses_star: True latent SES (standardized)
        - ses_obs: Observed SES proxy with measurement error
        - ses_quintile: Categorical SES (0-4 quintiles)
        - bmi: Body mass index with quadratic SES relationship
        - sbp: Systolic blood pressure with saturating SES effect
        - smoke: Binary smoking indicator with threshold effect
        - y: Binary disease outcome
        - s: Binary selection/participation indicator
        - p_s: Participation probability
        - p_y: Disease probability
        
    Notes
    -----
    Key relationships implemented:
    
    1. **SES Generation**: SES* ~ N(0.15*(50-age) + 0.15*sex + 0.25*PC1, 1)
    2. **Measurement Error**: SES_obs = 0.8*SES* + N(0, σ²)
    3. **BMI (U-shaped)**: BMI ~ 27.5 + 1.2*(SES*-0.2)² + noise
    4. **Smoking (threshold)**: Higher rates below SES* = -0.5
    5. **Blood Pressure (saturating)**: Improvement plateaus at high SES
    6. **Disease**: Direct SES effect plus mediation through risk factors
    7. **Selection**: Healthy volunteer bias with SES-dependent participation
    
    The simulation produces realistic epidemiological patterns including:
    - Non-linear dose-response relationships
    - Measurement error attenuation
    - Selection bias favoring healthier, higher-SES participants
    
    Examples
    --------
    >>> params = DGMParams(n=50000, seed=123)
    >>> pop_data = simulate_population(params)
    >>> print(f"Population size: {len(pop_data)}")
    >>> print(f"Disease prevalence: {pop_data['y'].mean():.1%}")
    >>> print(f"Participation rate: {pop_data['s'].mean():.1%}")
    """
    rng = np.random.default_rng(params.seed)
    n = params.n

    # Baseline covariates C
    age = rng.normal(60, 10, n).clip(40, 85)
    sex = rng.integers(0, 2, n)  # 0=female, 1=male
    pc1 = rng.normal(0, 1, n)

    # Latent SES*
    ses_star = rng.normal(loc=0.15 * standardize(50 - age) + 0.15*sex + 0.25*pc1, scale=1.0, size=n)

    # Observed SES proxy (noisy, then optionally coarsened)
    ses_obs = 0.8 * ses_star + rng.normal(0, params.ses_me_sd, n)

    # Simple quintiles (coarsened)
    q = pd.qcut(ses_obs, 5, labels=False)  # 0..4
    ses_quintile = q.astype(int)

    # Mediators (non-linear)
    # BMI: quadratic (U-ish), lower with mid SES*
    bmi = 27.5 + 1.2*(ses_star - 0.2)**2 + 0.05*(age-60) + 0.6*sex + rng.normal(0, 2.5, n)

    # Smoking: logistic with threshold and SES*
    lp_smoke = -0.5 - 1.2*ses_star + 0.6*(ses_star < -0.5).astype(float) + 0.01*(age-60) + 0.2*sex
    p_smoke = logistic(lp_smoke)
    smoke = rng.binomial(1, p_smoke, n)

    # SBP: saturating improvement with SES*
    sbp = 125 - 4*np.log1p(np.exp(ses_star - 0.0)) + 0.12*(age-60) + 3.0*sex + rng.normal(0, 8.0, n)

    # Outcome Y (binary disease within timeframe)
    # Direct SES* (mild monotone + threshold), plus mediators, plus C
    g_ses = 0.25*ses_star + 0.15*(ses_star < -0.8).astype(float)
    lp_y = (params.y_intercept + g_ses
            + 0.06*(sbp-120) + 0.08*(bmi-25) + 1.0*smoke
            + 0.018*(age-60) + 0.12*sex)

    p_y = logistic(lp_y)
    y = rng.binomial(1, p_y, n)

    # Selection S (biobank participation)
    # Use an approximate risk score (without intercept) as a driver of selection
    risk_score = (g_ses + 0.06*(sbp-120) + 0.08*(bmi-25) + 1.0*smoke + 0.018*(age-60) + 0.12*sex)
    lp_s = (params.sel_intercept + params.sel_ses*ses_star
            + params.sel_smoke*smoke
            + params.sel_bmi*(bmi-25)
            + params.sel_age*standardize(age)
            + params.sel_risk_score*risk_score)
    p_s = logistic(lp_s)
    s = rng.binomial(1, p_s, n)

    df = pd.DataFrame({
        "age": age,
        "sex": sex,
        "pc1": pc1,
        "ses_star": ses_star,
        "ses_obs": ses_obs,
        "ses_quintile": ses_quintile,
        "bmi": bmi,
        "sbp": sbp,
        "smoke": smoke,
        "y": y,
        "s": s,
        "p_s": p_s,
        "p_y": p_y
    })
    return df

def sample_biobank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the biobank sample from population data based on selection indicator.
    
    Filters the population dataset to include only individuals who would
    participate in a biobank study (s=1), creating a selected sample that
    exhibits systematic bias relative to the target population.
    
    Parameters
    ----------
    df : pd.DataFrame
        Population dataset containing selection indicator 's'.
        
    Returns
    -------
    pd.DataFrame
        Biobank sample containing only participants (s=1).
        Typically 5-15% of the original population size.
        
    Notes
    -----
    The biobank sample will exhibit:
    - Higher mean SES (selection bias)
    - Lower disease prevalence (healthy volunteer bias)
    - Different covariate distributions
    - Potential collider bias when conditioning on selection
    
    This function simulates the real-world scenario where biobank participants
    are not representative of the general population, creating challenges for
    causal inference and generalizability.
    
    Examples
    --------
    >>> pop_data = simulate_population(DGMParams(n=100000))
    >>> biobank_data = sample_biobank(pop_data)
    >>> print(f"Selection ratio: {len(biobank_data)/len(pop_data):.1%}")
    >>> print(f"SES bias: {biobank_data['ses_star'].mean():.2f} vs {pop_data['ses_star'].mean():.2f}")
    """
    return df.loc[df["s"]==1].copy()

# -----------------
# Analytic models
# -----------------

def fit_logit(formula: str, data: pd.DataFrame, weights: Optional[np.ndarray] = None) -> Any:
    """
    Fit a logistic regression model using statsmodels formula interface.
    
    Convenience wrapper for fitting binary logistic regression models with
    optional frequency weights, commonly used in epidemiological analyses.
    
    Parameters
    ----------
    formula : str
        Model formula in patsy/R syntax (e.g., 'y ~ x1 + x2 + bs(x3, df=5)').
        Supports transformations, interactions, and spline terms.
    data : pd.DataFrame
        Dataset containing variables referenced in the formula.
    weights : array-like, optional
        Frequency weights for weighted logistic regression. Default is None.
        
    Returns
    -------
    fitted model object
        Fitted logistic regression results object with methods for prediction,
        summary statistics, and model diagnostics.
        
    Examples
    --------
    >>> # Simple logistic regression
    >>> results = fit_logit('y ~ x1 + x2', data)
    >>> 
    >>> # With spline terms
    >>> results = fit_logit('y ~ bs(ses, df=5) + age + sex', data)
    >>> 
    >>> # With weights
    >>> results = fit_logit('y ~ x1 + x2', data, weights=sample_weights)
    """
    model = sm.GLM.from_formula(formula=formula, data=data, family=sm.families.Binomial(), freq_weights=weights)
    res = model.fit()
    return res

def null_llf(y: np.ndarray) -> float:
    """
    Calculate log-likelihood of the null (intercept-only) model for binary outcomes.
    
    Computes the maximum likelihood under the null hypothesis that all observations
    have the same probability of success, estimated as the sample proportion.
    This serves as a baseline for calculating pseudo-R² measures.
    
    Parameters
    ----------
    y : array-like
        Binary outcome variable (0s and 1s).
        
    Returns
    -------
    float
        Log-likelihood of the null model: LL₀ = Σᵢ [yᵢlog(p̂) + (1-yᵢ)log(1-p̂)]
        where p̂ is the sample proportion of successes.
        
    Notes
    -----
    The null model probability is clipped to [1e-6, 1-1e-6] to avoid numerical
    issues with log(0) when all outcomes are 0 or 1.
    
    This function is essential for computing McFadden's pseudo-R² and related
    goodness-of-fit measures for logistic regression models.
    
    Examples
    --------
    >>> y = np.array([0, 1, 0, 1, 1])
    >>> ll_null = null_llf(y)
    >>> print(f"Null log-likelihood: {ll_null:.3f}")
    """
    p = np.clip(np.mean(y), 1e-6, 1-1e-6)
    ll = np.sum(y*np.log(p) + (1-y)*np.log(1-p))
    return float(ll)

def g_compute_dose_response(model: Any, data: pd.DataFrame, ses_var: str, a_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate m(a) = E[Y | do(SES= a)] by setting SES variable to a and predicting with the fitted model,
    averaging over the empirical distribution of observed confounders in `data`.
    Note: This treats `ses_var` as the manipulable exposure in the fitted model (could be SES* oracle or SES_obs proxy).
    """
    base = data.copy()
    means = []
    for a in a_grid:
        tmp = base.copy()
        tmp[ses_var] = a
        # For splines, the design matrix updates automatically via formula interface.
        p = model.predict(tmp)
        means.append(np.mean(p))
    return a_grid, np.array(means)

def true_dose_response(df_population: pd.DataFrame, a_grid: np.ndarray, seed: int=7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the true m(a) = E[Y | do(SES*=a)] by re-simulating mediators and outcome
    while fixing SES* := a and integrating over exogenous noise and baseline covariates.
    """
    rng = np.random.default_rng(seed)
    age = df_population["age"].to_numpy()
    sex = df_population["sex"].to_numpy()
    pc1 = df_population["pc1"].to_numpy()

    means = []
    n = len(df_population)
    for a in a_grid:
        # mediators under intervention
        bmi = 27.5 + 1.2*(a - 0.2)**2 + 0.05*(age-60) + 0.6*sex + rng.normal(0, 2.5, n)
        lp_smoke = -0.5 - 1.2*a + 0.6*(a < -0.5) + 0.01*(age-60) + 0.2*sex
        p_smoke = 1.0 / (1.0 + np.exp(-lp_smoke))
        smoke = rng.binomial(1, p_smoke, n)
        sbp = 125 - 4*np.log1p(np.exp(a - 0.0)) + 0.12*(age-60) + 3.0*sex + rng.normal(0, 8.0, n)
        g_ses = 0.25*a + 0.15*(a < -0.8)
        lp_y = (-4.1 + g_ses
                + 0.06*(sbp-120) + 0.08*(bmi-25) + 1.0*smoke
                + 0.018*(age-60) + 0.12*sex)
        p_y = 1.0 / (1.0 + np.exp(-lp_y))
        means.append(np.mean(p_y))
    return a_grid, np.array(means)

def causal_R2_from_curve(m_a: np.ndarray, y: np.ndarray) -> float:
    """
    Causal R^2 (risk scale): Var_a( m(a) ) / Var(Y), where a ~ empirical SES distribution.
    Here we approximate Var_a(m(a)) by taking the variance across m(a) evaluated over
    SES grid weighted equally; you can reweight by SES density if desired.
    """
    var_y = np.var(y, ddof=0)
    var_ma = np.var(m_a, ddof=0)
    return float(var_ma / var_y)

# -----------------
# Demo / single-run analysis
# -----------------

def run_demo(n_pop: int = 200_000, seed: int = 42) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any, Any, Any, Any]:
    """
    Execute complete simulation study comparing SES-health causal inference methods.
    
    Runs a comprehensive analysis comparing traditional epidemiological approaches
    with modern causal inference methods for estimating socioeconomic effects on
    health outcomes in the presence of selection bias and measurement error.
    
    This function implements the full simulation pipeline:
    1. Generate population with complex SES-health relationships
    2. Apply biobank selection bias
    3. Fit multiple analytical models
    4. Estimate causal dose-response curves
    5. Compute performance metrics and decompositions
    
    Parameters
    ----------
    n_pop : int, default=200_000
        Population size for simulation. Larger values provide more stable
        estimates but increase computational time.
    seed : int, default=42
        Random seed for reproducible results.
        
    Returns
    -------
    Tuple containing:
        results : Dict[str, float]
            Performance metrics including:
            - Sample sizes and prevalence rates
            - McFadden and Tjur R² for different models
            - Causal R² estimates (true, oracle, proxy)
            - Shapley value decomposition
            
        pop : pd.DataFrame
            Full population dataset
            
        biobank : pd.DataFrame  
            Selected biobank sample
            
        a_grid : np.ndarray
            SES* intervention grid
            
        a_grid_obs : np.ndarray
            SES_obs intervention grid
            
        m_true : np.ndarray
            True causal dose-response curve
            
        m_hat_star : np.ndarray
            Oracle G-computation estimates (using SES*)
            
        m_hat_obs : np.ndarray
            Proxy G-computation estimates (using SES_obs)
            
        res_lin, res_over, res_spline_obs, res_spline_star : fitted model objects
            Fitted regression models for further analysis
            
    Notes
    -----
    The analysis compares:
    
    **Traditional Approaches (Problematic):**
    - Linear quintile models (mis-specification)
    - Over-adjusted models including mediators (collider bias)
    
    **Modern Approaches (Preferred):**
    - Flexible spline models capturing non-linearity
    - G-computation for causal dose-response curves
    - Oracle vs. proxy SES comparisons
    
    **Key Metrics:**
    - McFadden's/Tjur's R²: Traditional model fit
    - Causal R²: Proportion of variance from SES interventions
    - Shapley values: Fair attribution of explanatory power
    
    The simulation demonstrates:
    - Bias from selection and measurement error
    - Advantages of flexible modeling approaches
    - Importance of causal vs. predictive frameworks
    
    Examples
    --------
    >>> # Run basic demo
    >>> results, pop, biobank, *curves = run_demo(n_pop=50000)
    >>> print(f"True causal R²: {results['causal_R2_true']:.3f}")
    >>> print(f"Biobank bias: {results['mean_y_population']:.3f} vs {results['mean_y_biobank']:.3f}")
    >>> 
    >>> # Custom parameters
    >>> results, *_ = run_demo(n_pop=100000, seed=456)
    >>> 
    >>> # Extract dose-response curves for plotting
    >>> _, _, _, a_grid, a_grid_obs, m_true, m_oracle, m_proxy, *_ = run_demo()
    >>> plt.plot(a_grid, m_true, label='True curve')
    >>> plt.plot(a_grid, m_oracle, label='Oracle estimate')
    >>> plt.plot(a_grid_obs, m_proxy, label='Proxy estimate')
    """
    params = DGMParams(n=n_pop, seed=seed)
    pop = simulate_population(params)
    biobank = pop.loc[pop["s"]==1].copy()

    # SES grids
    a_grid = np.linspace(np.percentile(pop["ses_star"], 2),
                         np.percentile(pop["ses_star"], 98), 30)
    a_grid_obs = np.linspace(np.percentile(pop["ses_obs"], 2),
                             np.percentile(pop["ses_obs"], 98), 30)

    # True curve (population, oracle)
    _, m_true = true_dose_response(pop, a_grid)

    # A) Mis-specified models (selected sample)
    # Linear SES quintile (treated numeric) + C
    formula_lin = "y ~ ses_quintile + age + sex + pc1"
    res_lin = sm.GLM.from_formula(formula_lin, data=biobank, family=sm.families.Binomial()).fit()
    p_lin = res_lin.predict(biobank)
    r2_lin_mcf = mc_fadden_r2(res_lin.llf, null_llf(biobank["y"].to_numpy()))
    r2_lin_tjur = tjur_r2(biobank["y"], p_lin)

    # Over-adjustment (include mediators; typical risk-factor model)
    formula_over = "y ~ ses_quintile + age + sex + pc1 + bmi + sbp + smoke"
    res_over = sm.GLM.from_formula(formula_over, data=biobank, family=sm.families.Binomial()).fit()
    p_over = res_over.predict(biobank)
    r2_over_mcf = mc_fadden_r2(res_over.llf, null_llf(biobank["y"].to_numpy()))

    # B) Flexible spline on observed SES
    biobank = biobank.assign(
        ses_obs_spline = biobank["ses_obs"]  # keep name for formula
    )
    formula_spline_obs = "y ~ bs(ses_obs_spline, df=5) + age + sex + pc1"
    res_spline_obs = sm.GLM.from_formula(formula_spline_obs, data=biobank, family=sm.families.Binomial()).fit()

    # (Oracle) Flexible spline on SES*
    biobank = biobank.assign(
        ses_star_spline = biobank["ses_star"]
    )
    formula_spline_star = "y ~ bs(ses_star_spline, df=5) + age + sex + pc1"
    res_spline_star = sm.GLM.from_formula(formula_spline_star, data=biobank, family=sm.families.Binomial()).fit()

    # C) G-computation curves (selected sample)
    _, m_hat_obs = g_compute_dose_response(res_spline_obs, biobank, "ses_obs_spline", a_grid_obs)
    _, m_hat_star = g_compute_dose_response(res_spline_star, biobank, "ses_star_spline", a_grid)

    # Causal R^2 (approx) using curves; compare to observed Var(Y)
    r2_causal_true = causal_R2_from_curve(m_true, pop["y"].to_numpy())
    r2_causal_oracle = causal_R2_from_curve(m_hat_star, biobank["y"].to_numpy())
    r2_causal_proxy = causal_R2_from_curve(m_hat_obs, biobank["y"].to_numpy())

    # Simple two-group Shapley (SES vs others) on selected sample
    # Group 1: SES* spline (oracle); Group 2: C = age, sex, pc1
    llf_null = null_llf(biobank["y"].to_numpy())
    llf_g1 = sm.GLM.from_formula("y ~ bs(ses_star_spline, df=5)", data=biobank, family=sm.families.Binomial()).fit().llf
    llf_g2 = sm.GLM.from_formula("y ~ age + sex + pc1", data=biobank, family=sm.families.Binomial()).fit().llf
    llf_both = res_spline_star.llf
    shap = shapley_two_groups_r2(llf_null, llf_g1, llf_g2, llf_both)

    out = {
        "n_population": int(len(pop)),
        "n_biobank": int(len(biobank)),
        "mean_y_population": float(pop["y"].mean()),
        "mean_y_biobank": float(biobank["y"].mean()),
        "mcFadden_R2_linear_quintile": float(r2_lin_mcf),
        "Tjur_R2_linear_quintile": float(r2_lin_tjur),
        "mcFadden_R2_overadjusted": float(r2_over_mcf),
        "causal_R2_true": float(r2_causal_true),
        "causal_R2_oracle_SESstar": float(r2_causal_oracle),
        "causal_R2_proxy_SESobs": float(r2_causal_proxy),
        "Shapley_SESstar_share": shap["G1_share"],
        "Shapley_C_share": shap["G2_share"],
        "Shapley_R2_total": shap["R2_total"],
    }
    return out, pop, biobank, a_grid, a_grid_obs, m_true, m_hat_star, m_hat_obs, res_lin, res_over, res_spline_obs, res_spline_star


# -----------------
# Main execution
# -----------------

if __name__ == "__main__":
    """
    Run the SES simulation demo when script is executed directly.
    """
    print("SES Simulation Scaffold - Running Demo Analysis")
    print("=" * 50)
    
    # Run the simulation
    print("Generating population and running analysis...")
    results, pop, biobank, a_grid, a_grid_obs, m_true, m_hat_star, m_hat_obs, *models = run_demo(
        n_pop=50_000,  # Smaller sample for faster demo
        seed=42
    )
    
    print("\n" + "Results Summary" + "\n" + "-" * 20)
    
    # Sample sizes and basic stats
    print(f"Population size: {results['n_population']:,}")
    print(f"Biobank sample: {results['n_biobank']:,} ({results['n_biobank']/results['n_population']:.1%})")
    print(f"Population disease prevalence: {results['mean_y_population']:.1%}")
    print(f"Biobank disease prevalence: {results['mean_y_biobank']:.1%}")
    
    # Model performance comparison
    print(f"\nModel Performance (McFadden R²):")
    print(f"  Linear quintile model: {results['mcFadden_R2_linear_quintile']:.3f}")
    print(f"  Over-adjusted model: {results['mcFadden_R2_overadjusted']:.3f}")
    print(f"  Tjur R² (linear model): {results['Tjur_R2_linear_quintile']:.3f}")
    
    # Causal R² estimates
    print(f"\nCausal R² Estimates:")
    print(f"  True causal R²: {results['causal_R2_true']:.3f}")
    print(f"  Oracle estimate (SES*): {results['causal_R2_oracle_SESstar']:.3f}")
    print(f"  Proxy estimate (SES_obs): {results['causal_R2_proxy_SESobs']:.3f}")
    
    # Shapley decomposition
    print(f"\nShapley Value Decomposition:")
    print(f"  SES* contribution: {results['Shapley_SESstar_share']:.3f}")
    print(f"  Other covariates: {results['Shapley_C_share']:.3f}")
    print(f"  Total R²: {results['Shapley_R2_total']:.3f}")
    
    # Interpretation
    print(f"\n" + "Key Insights" + "\n" + "-" * 15)
    
    # Selection bias
    bias_direction = "healthier" if results['mean_y_biobank'] < results['mean_y_population'] else "sicker"
    bias_magnitude = abs(results['mean_y_biobank'] - results['mean_y_population']) / results['mean_y_population']
    print(f"• Biobank shows {bias_magnitude:.1%} {bias_direction} population (healthy volunteer bias)")
    
    # Causal vs predictive
    oracle_vs_true = results['causal_R2_oracle_SESstar'] / results['causal_R2_true']
    proxy_vs_true = results['causal_R2_proxy_SESobs'] / results['causal_R2_true']
    print(f"• Oracle method captures {oracle_vs_true:.1%} of true causal effect")
    print(f"• Proxy method captures {proxy_vs_true:.1%} of true causal effect")
    
    # SES importance
    ses_importance = results['Shapley_SESstar_share'] / results['Shapley_R2_total']
    print(f"• SES accounts for {ses_importance:.1%} of total explainable variance")
    
    print(f"\n" + "Methodological Lessons" + "\n" + "-" * 25)
    print("• Selection bias reduces disease prevalence in biobank samples")
    print("• Measurement error in SES attenuates causal effect estimates") 
    print("• Flexible modeling approaches better capture non-linear relationships")
    print("• G-computation provides interpretable causal dose-response curves")
    
    print(f"\nDemo completed successfully!")
    print("Tip: Use the returned objects for further analysis and visualization.")
