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
import pandas as pd
from IPython.display import display, Markdown


m = pd.read_csv("../artifacts/simulation_metrics.csv").iloc[0].to_dict()

n_pop       = int(m.get("n_population", 0))
n_bio       = int(m.get("n_biobank", 0))
prev_pop    = float(m.get("mean_y_population", float("nan")))
prev_bio    = float(m.get("mean_y_biobank", float("nan")))
r2_lin      = float(m.get("mcFadden_R2_linear_quintile", float("nan")))
r2_over     = float(m.get("mcFadden_R2_overadjusted", float("nan")))
r2_true     = float(m.get("causal_R2_true", float("nan")))
r2_oracle   = float(m.get("causal_R2_oracle_SESstar", float("nan")))
r2_proxy    = float(m.get("causal_R2_proxy_SESobs", float("nan")))

sel_frac = (n_bio / n_pop) if n_pop else float("nan")

output = display(Markdown( **Background.** Socioeconomic status (SES) is often modeled linearly or as coarse categories in large cohorts. When true SES→risk relationships are non-linear and samples are selected (healthy-volunteer bias), SES’s role can be under-estimated.

**Methods.** We simulated a latent SES* affecting mediators of cardiovascular disease(BMI, systolic BP, smoking) via non-linear functions and directly affecting a binary outcome. A “biobank-like” sample was generated via selection favoring higher SES and lower risk. We compared typical models (linear/quintile SES; with/without mediator adjustment) to spline models with g-computation of E[Y | do(SES=a)]. We summarized SES attribution via a causal variance share (R^2 causal) and a two-block Shapley split.

**Results.** Selection yielded a biobank fraction of **{sel_frac:.1%}** (N={n_bio:,}/{n_pop:,}) and reduced prevalence from **{prev_pop:.2%}** (population) to **{prev_bio:.2%}** (selected). In the selected sample, linear-quintile SES achieved **McFadden R^{2} ={r2_lin:.3f}**; including mediators raised predictive fit (**{r2_over:.3f}**) while down-weighting SES as a putative cause. The **causal R^{2}** was **{r2_true:.4f}** in the population; within the selected sample it was **{r2_oracle:.4f}** using oracle SES* and **{r2_proxy:.4f}** using a noisy proxy.

**Conclusions.** Selection plus functional-form misspecification materially underestimates SES’s contribution. Flexible modeling (splines/GAMs) with standardization (or TMLE) restores more of SES’s causal role and should be preferred in large cohorts.
))

```
#
#
#
#
