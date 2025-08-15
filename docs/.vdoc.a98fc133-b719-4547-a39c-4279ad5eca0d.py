# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: false
#| asis: true
import pandas as pd, numpy as np, os

def read_or_none(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

m_sim   = read_or_none("../artifacts/simulation_metrics.csv")
chg_sim = read_or_none("../artifacts/causal_r2_changes.csv")

m_pop   = read_or_none("../artifacts/nhanes_population_metrics.csv")
m_sel   = read_or_none("../artifacts/nhanes_biobanklike_metrics.csv")
dr_pop  = read_or_none("../artifacts/nhanes_dose_response_population.csv")
dr_sel  = read_or_none("../artifacts/nhanes_dose_response_biobanklike.csv")

def get_sim_val(key, default=float("nan")):
    try:
        return float(m_sim.iloc[0].get(key, default))
    except Exception:
        return default

def pct_text(x):
    return "NA" if (x is None or not np.isfinite(x)) else f"{x:.0f}%"
def ratio_text(x):
    return "NA" if (x is None or not np.isfinite(x)) else f"{x:.2f}"

# --- Simulation numbers ---
n_pop      = int(get_sim_val("n_population", 0))
n_bio      = int(get_sim_val("n_biobank",   0))
prev_pop   = get_sim_val("mean_y_population")
prev_sel   = get_sim_val("mean_y_biobank")
r2_lin     = get_sim_val("mcFadden_R2_linear_quintile")
r2_over    = get_sim_val("mcFadden_R2_overadjusted")

# Causal R2 % changes
def pull_contrast(df, cid):
    if df is None: return (None, None)
    row = df.loc[df["contrast_id"]==cid]
    if len(row)==0: return (None, None)
    return float(row["pct_change"].iloc[0]), float(row["ratio"].iloc[0])

pc_pop_to_sel_oracle, ra_pop_to_sel_oracle = pull_contrast(chg_sim, "pop_to_sel_oracle")
pc_sel_oracle_to_proxy, ra_sel_oracle_to_proxy = pull_contrast(chg_sim, "sel_oracle_to_sel_proxy")

# --- NHANES analog causal R2 using spline dose-response ---
def causal_r2_from_curve(curve_df, prev):
    try:
        arr = np.asarray(curve_df["risk"].values, dtype=float)
        w = np.ones_like(arr)/len(arr)
        mu = float(np.sum(w*arr))
        num = float(np.sum(w*(arr-mu)**2))
        varY = float(prev)*(1-float(prev))
        return num/varY if varY>0 else float("nan")
    except Exception:
        return float("nan")

nh_prev_pop = float(m_pop.iloc[0]["prev"]) if m_pop is not None and "prev" in m_pop.columns else float("nan")
nh_prev_sel = float(m_sel.iloc[0]["prev"]) if m_sel is not None and "prev" in m_sel.columns else float("nan")

nh_R2_pop = causal_r2_from_curve(dr_pop, nh_prev_pop) if dr_pop is not None else float("nan")
nh_R2_sel = causal_r2_from_curve(dr_sel, nh_prev_sel) if dr_sel is not None else float("nan")

nh_ratio = (nh_R2_sel/nh_R2_pop) if (nh_R2_pop and np.isfinite(nh_R2_pop) and nh_R2_pop!=0) else float("nan")
nh_pct   = 100.0*(nh_R2_sel - nh_R2_pop)/nh_R2_pop if (nh_R2_pop and np.isfinite(nh_R2_pop) and nh_R2_pop!=0) else float("nan")

display(Markdown("""
**Background.** Socioeconomic status (SES) is often modeled linearly or as coarse categories in large cohorts. When true SES&rarr;risk relationships are non-linear and samples are selected (healthy volunteer bias), SES's role can be under-estimated.

**Methods.** We built a DAG‑informed simulation with latent (SES^{{*}}) affecting mediators (BMI, systolic BP, smoking) via non‑linear functions and directly affecting a binary outcome. We generated a “biobank‑like” sample by preferentially selecting higher (SES^{{*}}) and lower risk. We compared typical models (linear/quintile SES; with/without mediator adjustment) to splines with g‑computation of (E[Y mid do(SES=a)]). We summarized SES attribution via a causal variance share (R^2{{text{{causal}}}}) and a two‑block Shapley split.

**Results (simulation).** Selection yielded a biobank fraction of **{(n_bio/n_pop):.1%}** ((N={n_bio:,}/{n_pop:,})) and reduced prevalence from **{prev_pop:.2%}** (population) to **{prev_sel:.2%}** (selected). In the selected sample, linear‑quintile SES achieved **McFadden (R^2={r2_lin:.3f})**; including mediators raised predictive fit (**{r2_over:.3f}**) while down‑weighting SES as a putative cause. The SES causal (R^2) **changed by {pct_text(pc_pop_to_sel_oracle)}** (ratio {ratio_text(ra_pop_to_sel_oracle)}) from population to selected (oracle (SES^{{*}})), and by **{pct_text(pc_sel_oracle_to_proxy)}** (ratio {ratio_text(ra_sel_oracle_to_proxy)}) when replacing oracle with a noisy proxy in the selected sample. In the NHANES case study using NHANES 2003–2018 (DEMO/BPX/BMX/MCQ/SMQ), spline + g‑computation estimated an analog of SES causal (R^2) of **{nh_R2_pop:.4f}** (survey‑weighted population) and **{nh_R2_sel:.4f}** (biobank‑like selection), a **{pct_text(nh_pct)}** change (ratio {ratio_text(nh_ratio)}).

**Conclusions.** Across simulation and NHANES, **selection plus functional‑form misspecification underestimates SES’s contribution**. Flexible SES modeling (splines/GAMs) with standardization (or TMLE) recovers more of the causal signal and should be preferred in large cohorts.
"""))
##print(txt)
```
#
#
#
#
