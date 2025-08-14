# Simulation of Residual Confounding by SES
_A DAG-informed simulation + NHANES case-study scaffold_

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.TBD.svg)](https://doi.org/10.5281/zenodo.TBD)

## How to cite
If you use this repository, please cite the archived release (Zenodo DOI above). Example (replace `TBD` with your DOI):

**BibTeX**
```bibtex
@software{yourname_ses_residual_confounding_2025,
  author  = {Dunne, R.A},
  title   = {Simulation of Residual Confounding by SES},
  year    = {2025},
  publisher = {Zenodo},
  version = {v1.0.0},
  doi     = {10.5281/zenodo.TBD},
  url     = {https://doi.org/10.5281/zenodo.TBD}
}
```

## Summary
This project investigates how **non-linear SES→risk→disease** relationships and **selection (healthy-volunteer) bias** can **underestimate** the causal role of SES and its variance share in large cohorts. We provide:

- **Part A — Simulation study** (DAG-informed) with latent SES*, non-linear mediators, selection, and measurement error; causal and predictive estimators; variance attribution.
- **Part B — NHANES scaffold** for a comparable real-data analysis (runs if NHANES XPT files are available locally; not distributed here).

Outputs: dose–response figures, pseudo-/causal \(R^2\), and compact result tables.

## Repository structure
```
SES_residual_confounding_reproducible_v2.ipynb  # Parameterized notebook (papermill-ready), inline tests, styled figures
Makefile                                        # run / test / html / clean
requirements.txt                                # pip requirements (alt to conda)
environment.yml                                 # conda environment (recommended)
tests/
  └── test_sim.py                               # Pytest invariants for the simulator
ses_sim_new.py                                  # Your refined simulation scaffold (importable)
artifacts/                                      # (created at runtime) metrics, figures, module copy, session info
```

## Quick start (Conda, recommended)
```bash
# 1) Create and activate environment
conda env create -f environment.yml
conda activate ses-confounding

# 2) Execute the notebook with papermill (writes artifacts/)
make run

# 3) Run tests (after notebook creates artifacts/ses_sim_module.py)
make test

# 4) Export executed notebook to HTML
make html
```

### Alternative (pip)
```bash
python -m venv .venv && source .venv/bin/activate  # or use your preferred tooling
pip install -r requirements.txt
make run && make test && make html
```

## Reproducibility checklist
- **Parameterised notebook** (papermill) with a `parameters` cell (population size, seed, artifact path, optional NHANES paths).
- **Deterministic seeds** in key steps.
- **Environment capture**: `artifacts/session_info.json` (Python & package versions + code hashes).
- **Inline tests**: basic invariants on prevalence ranges, curve shapes, and \(R^2\) bounds.
- **Raw artifacts**: CSV metrics and PNG figures emitted under `artifacts/`.

## NHANES data availability
NHANES XPT files are **not** redistributed. To run Part B, point the notebook to your local NHANES directory:
```python
NHANES_BASE_DIR = "/path/to/nhanes_xpt"
NHANES_CYCLES   = ["C","D","E","F","G","H","I","J"]  # 2003–2018; adjust as needed
```
The NHANES code **skips gracefully** if files are absent.

## License
**MIT License** (recommended for open reproducibility). If you need a different license, update `LICENSE` and the badge/metadata before minting a DOI.

## Versioning & releases
- Tag releases as `vMAJOR.MINOR.PATCH`.
- Sync a Zenodo archive at each release to obtain a DOI.
- Update the DOI badge and the citation block after the first release.

## Acknowledgements
Thanks to collaborators and institutions supporting this work. (Add grants and affiliations here.)

---

### Changelog
- **v1.0.0** — Initial public release with parameterised notebook, Makefile, tests, and NHANES scaffold.
