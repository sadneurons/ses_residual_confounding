
import importlib.util, sys, os, numpy as np, pandas as pd

# Load the module copy written by the notebook when executed
mod_path = os.path.join("artifacts","ses_sim_module.py")
if not os.path.exists(mod_path):
    # Fallback: try project root's ses_sim_new.py if available
    mod_path = os.path.join(os.getcwd(), "ses_sim_new.py")

spec = importlib.util.spec_from_file_location("ses_sim_module", mod_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def test_run_demo_shapes():
    res, pop, biobank, a_grid, a_grid_obs, m_true, m_hat_star, m_hat_obs, *_ = mod.run_demo(n_pop=50_000, seed=7)
    assert isinstance(res, dict)
    assert len(pop) >= 40_000
    assert len(biobank) > 0
    assert len(a_grid) == len(m_true)
    assert 0 <= pop['y'].mean() <= 1
    assert 0 <= biobank['y'].mean() <= 1

def test_causal_r2_bounds():
    res, *_ = mod.run_demo(n_pop=30_000, seed=11)
    for k in ['causal_R2_true','causal_R2_oracle_SESstar','causal_R2_proxy_SESobs']:
        assert 0.0 <= res[k] <= 0.2
