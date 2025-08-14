
# Makefile for SES residual confounding project
# Usage:
#   make env        # show environment
#   make run        # execute notebook with papermill
#   make test       # run pytest tests
#   make html       # export executed notebook to HTML
#   make clean      # remove artifacts

NB=SES_residual_confounding_reproducible_v2.ipynb
OUT=SES_residual_confounding_executed.ipynb
HTML=SES_residual_confounding_executed.html

env:
	python -V
	pip list | grep -E 'numpy|pandas|statsmodels|patsy|matplotlib|papermill|nbconvert|pytest' || true

run:
	papermill $(NB) $(OUT) -p N_POP 200000 -p SEED 123 -p ART_DIR artifacts

test:
	pytest -q

html: run
	jupyter nbconvert --to html $(OUT) --output $(HTML)

clean:
	rm -rf artifacts __pycache__ *.html *.ipynb_checkpoint

# --- Document builds ---
.PHONY: abstract poster docs abstract-md

# Build the PDF abstract (requires quarto + tinytex)
abstract: run
	quarto render docs/abstract.qmd --to pdf

# Build the HTML poster  
poster: run
	quarto render docs/poster.qmd --to html

# Alternative: convert to simple markdown (no Quarto needed)
abstract-md: run
	python -c "import pandas as pd; m=pd.read_csv('artifacts/simulation_metrics.csv').iloc[0].to_dict(); print('# SES Simulation Results\n\n- Population N:', int(m.get('n_population',0)), '\n- Selected N:', int(m.get('n_biobank',0)), '\n- Causal R²:', f'{m.get(\"causal_R2_true\", 0):.4f}')" > docs/abstract_simple.md

# Convenience: try Quarto first, fallback to simple markdown
docs: run
	-quarto render docs/abstract.qmd --to pdf || echo "PDF generation failed - Quarto/Tectonic not available"
	-quarto render docs/poster.qmd --to html || echo "HTML poster generation failed - Quarto not available"